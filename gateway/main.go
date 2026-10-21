package main

// C 端接入层：限流、并发保护、带 TTL 的幂等、熔断与 HTTP/SSE 代理。
import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type breaker struct {
	failures atomic.Int64
	opened   atomic.Int64
	probe    atomic.Bool
}

func (b *breaker) allow() bool {
	opened := b.opened.Load()
	if opened == 0 {
		return true
	}
	return time.Now().UnixNano() > opened && b.probe.CompareAndSwap(false, true)
}
func (b *breaker) success() {
	b.failures.Store(0)
	b.opened.Store(0)
	b.probe.Store(false)
}
func (b *breaker) fail() {
	if b.failures.Add(1) >= 5 {
		b.opened.Store(time.Now().Add(10 * time.Second).UnixNano())
	}
	b.probe.Store(false)
}

func (b *breaker) rpcError(err error) {
	switch status.Code(err) {
	case codes.Internal, codes.Unavailable, codes.DeadlineExceeded, codes.Unknown, codes.DataLoss:
		b.fail()
	case codes.Canceled:
		b.probe.Store(false)
	default:
		b.success()
	}
}

type idemRecord struct {
	fingerprint string
	status      int
	contentType string
	body        []byte
	expiresAt   time.Time
}

type idemStore struct {
	mu       sync.Mutex
	ttl      time.Duration
	items    map[string]idemRecord
	inflight map[string]idemLease
	redis    *redisClient
	required bool
}

type idemLease struct {
	fingerprint string
	owner       string
}

func newIdemStore(ttl time.Duration, redisURL string) *idemStore {
	store := &idemStore{ttl: ttl, items: make(map[string]idemRecord), inflight: make(map[string]idemLease), required: redisURL != ""}
	if redisURL != "" {
		if client, err := newRedisClient(redisURL, 32); err == nil {
			store.redis = client
		}
	}
	return store
}

func (s *idemStore) ready() bool {
	if !s.required {
		return true
	}
	if s.redis == nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	value, err := s.redis.do(ctx, "PING")
	return err == nil && value == "PONG"
}

func (s *idemStore) get(key, fingerprint string) (idemRecord, bool, bool) {
	if s.redis != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		defer cancel()
		value, err := s.redis.do(ctx, "GET", "math-agent:idem:"+key)
		if err != nil || value == nil {
			return idemRecord{}, false, false
		}
		var stored struct {
			Fingerprint string `json:"fingerprint"`
			Status      int    `json:"status"`
			ContentType string `json:"content_type"`
			Body        string `json:"body"`
		}
		if json.Unmarshal([]byte(value.(string)), &stored) != nil {
			return idemRecord{}, false, false
		}
		if stored.Fingerprint != fingerprint {
			return idemRecord{}, true, true
		}
		if stored.Status == 0 {
			return idemRecord{}, true, false
		}
		body, _ := base64.StdEncoding.DecodeString(stored.Body)
		return idemRecord{fingerprint: stored.Fingerprint, status: stored.Status, contentType: stored.ContentType, body: body}, true, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for k, item := range s.items {
		if !now.Before(item.expiresAt) {
			delete(s.items, k)
		}
	}
	item, ok := s.items[key]
	if !ok {
		return idemRecord{}, false, false
	}
	if item.fingerprint != fingerprint {
		return idemRecord{}, true, true
	}
	return item, true, false
}

func (s *idemStore) reserve(key, fingerprint, owner string) (bool, bool, bool, error) {
	if s.redis != nil {
		payload, _ := json.Marshal(map[string]any{"fingerprint": fingerprint, "owner": owner, "status": 0})
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		result, err := s.redis.do(ctx, "SET", "math-agent:idem:"+key, string(payload), "NX", "EX", strconv.Itoa(max(1, int(s.ttl.Seconds()))))
		if err != nil {
			return false, false, false, err
		}
		if err == nil && result == "OK" {
			return true, false, false, nil
		}
		_, found, conflict := s.get(key, fingerprint)
		if conflict {
			return false, true, false, nil
		}
		return false, false, found, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if item, ok := s.items[key]; ok {
		if item.fingerprint != fingerprint {
			return false, true, false, nil
		}
		return false, false, true, nil
	}
	if existing, ok := s.inflight[key]; ok {
		return false, existing.fingerprint != fingerprint, true, nil
	}
	s.inflight[key] = idemLease{fingerprint: fingerprint, owner: owner}
	return true, false, false, nil
}

func (s *idemStore) set(key, fingerprint, owner string, status int, contentType string, body []byte) {
	if s.redis != nil {
		payload, _ := json.Marshal(map[string]any{"fingerprint": fingerprint, "status": status, "content_type": contentType, "body": base64.StdEncoding.EncodeToString(body)})
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		defer cancel()
		script := `local v=redis.call('GET',KEYS[1]); if not v then return 0 end; local r=cjson.decode(v); if r.owner~=ARGV[1] then return 0 end; redis.call('SET',KEYS[1],ARGV[2],'EX',ARGV[3]); return 1`
		_, _ = s.redis.do(ctx, "EVAL", script, "1", "math-agent:idem:"+key, owner, string(payload), strconv.Itoa(max(1, int(s.ttl.Seconds()))))
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inflight[key].owner != owner {
		return
	}
	delete(s.inflight, key)
	s.items[key] = idemRecord{fingerprint: fingerprint, status: status, contentType: contentType, body: append([]byte(nil), body...), expiresAt: time.Now().Add(s.ttl)}
}

func (s *idemStore) release(key, owner string) {
	if s.redis != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		defer cancel()
		script := `local v=redis.call('GET',KEYS[1]); if not v then return 0 end; local r=cjson.decode(v); if r.owner==ARGV[1] and r.status==0 then return redis.call('DEL',KEYS[1]) end; return 0`
		_, _ = s.redis.do(ctx, "EVAL", script, "1", "math-agent:idem:"+key, owner)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inflight[key].owner == owner {
		delete(s.inflight, key)
	}
}

type gateway struct {
	client   *http.Client
	ready    *http.Client
	upstream string
	grpcConn *grpc.ClientConn
	useGRPC  bool
	sem      chan struct{}
	breakers sync.Map
	idem     *idemStore
}

func (g *gateway) breakerFor(key string) *breaker {
	v, _ := g.breakers.LoadOrStore(key, &breaker{})
	return v.(*breaker)
}

func requestKeys(c *gin.Context, body []byte) (string, string) {
	bodyHash := sha256.Sum256(body)
	fingerprint := hex.EncodeToString(bodyHash[:])
	identity := c.GetHeader("X-User-ID")
	if identity == "" {
		identity = c.GetHeader("Authorization")
	}
	clientKey := c.GetHeader("Idempotency-Key")
	if clientKey == "" {
		clientKey = fingerprint
	}
	if identity == "" {
		identity = c.ClientIP()
	}
	logical := c.Request.Method + "|" + c.Request.URL.Path + "|" + identity + "|" + c.GetHeader("X-Session-ID") + "|" + clientKey
	h := sha256.Sum256([]byte(logical))
	return hex.EncodeToString(h[:]), fingerprint
}

func (g *gateway) validateIdentity(c *gin.Context) bool {
	secret := os.Getenv("AUTH_SHARED_SECRET")
	require := strings.EqualFold(os.Getenv("REQUIRE_USER_IDENTITY"), "true") || secret != ""
	if !require {
		return true
	}
	userID := c.GetHeader("X-User-ID")
	sessionID := c.GetHeader("X-Session-ID")
	if userID == "" || sessionID == "" || userID == "anonymous" || sessionID == "anonymous" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "需要受信任的 user_id 和 session_id", "code": "IDENTITY_REQUIRED"})
		return false
	}
	if secret == "" {
		return true
	}
	signature := c.GetHeader("X-User-Signature")
	signer := hmac.New(sha256.New, []byte(secret))
	_, _ = signer.Write([]byte(userID))
	if signature == "" || !hmac.Equal([]byte(signature), []byte(hex.EncodeToString(signer.Sum(nil)))) {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "身份认证失败", "code": "IDENTITY_INVALID"})
		return false
	}
	return true
}

func (g *gateway) ask(c *gin.Context) {
	if !g.validateIdentity(c) {
		return
	}
	select {
	case g.sem <- struct{}{}:
		defer func() { <-g.sem }()
	default:
		c.JSON(http.StatusTooManyRequests, gin.H{"error": "请求排队已满", "code": "QUEUE_FULL"})
		return
	}
	body, err := io.ReadAll(io.LimitReader(c.Request.Body, 1<<20))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求体读取失败"})
		return
	}
	if !g.idem.ready() {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "幂等存储暂不可用", "code": "IDEMPOTENCY_STORE_UNAVAILABLE"})
		return
	}
	wantsStream := strings.Contains(c.GetHeader("Accept"), "text/event-stream") || c.Request.URL.Path == "/ask/stream"
	key, fingerprint := requestKeys(c, body)
	ownerBytes := make([]byte, 16)
	if _, err := rand.Read(ownerBytes); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"code": "IDEMPOTENCY_STORE_UNAVAILABLE"})
		return
	}
	owner := hex.EncodeToString(ownerBytes)
	if !wantsStream {
		if cached, found, conflict := g.idem.get(key, fingerprint); found {
			if conflict {
				c.JSON(http.StatusConflict, gin.H{"error": "同一 Idempotency-Key 对应了不同请求", "code": "IDEMPOTENCY_CONFLICT"})
				return
			}
			if cached.status == 0 {
				c.JSON(http.StatusConflict, gin.H{"code": "IDEMPOTENCY_IN_PROGRESS", "error": "相同请求正在处理中"})
				return
			}
			c.Data(cached.status, cached.contentType, cached.body)
			return
		}
		owned, conflict, _, reserveErr := g.idem.reserve(key, fingerprint, owner)
		if reserveErr != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"code": "IDEMPOTENCY_STORE_UNAVAILABLE"})
			return
		}
		if conflict {
			c.JSON(http.StatusConflict, gin.H{"error": "同一 Idempotency-Key 对应了不同请求", "code": "IDEMPOTENCY_CONFLICT"})
			return
		}
		if !owned {
			c.JSON(http.StatusConflict, gin.H{"error": "相同请求正在处理中", "code": "IDEMPOTENCY_IN_PROGRESS"})
			return
		}
		if owned {
			defer g.idem.release(key, owner)
		}
	}
	b := g.breakerFor("python-ai")
	if !b.allow() {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI 服务熔断恢复中", "code": "UPSTREAM_OPEN"})
		return
	}
	if g.useGRPC {
		g.askGRPC(c, body, key, fingerprint, owner, wantsStream)
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 45*time.Second)
	defer cancel()
	path := c.Request.URL.Path
	if path == "" {
		path = "/ask"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.upstream+path, &byteReader{b: body})
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求构造失败"})
		return
	}
	for _, header := range []string{"Content-Type", "Accept", "Authorization", "X-User-ID", "X-Session-ID", "X-User-Signature", "X-Request-ID"} {
		if value := c.GetHeader(header); value != "" {
			req.Header.Set(header, value)
		}
	}
	resp, err := g.client.Do(req)
	if err != nil {
		b.fail()
		c.JSON(http.StatusGatewayTimeout, gin.H{"error": "推理服务超时", "code": "UPSTREAM_TIMEOUT"})
		return
	}
	defer resp.Body.Close()
	if wantsStream || strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		c.Header("Content-Type", "text/event-stream; charset=utf-8")
		c.Header("Cache-Control", "no-cache")
		c.Header("X-Accel-Buffering", "no")
		c.Status(resp.StatusCode)
		f, _ := c.Writer.(http.Flusher)
		buf := make([]byte, 32768)
		for {
			n, readErr := resp.Body.Read(buf)
			if n > 0 {
				_, _ = c.Writer.Write(buf[:n])
				if f != nil {
					f.Flush()
				}
			}
			if readErr != nil {
				break
			}
		}
		if resp.StatusCode >= 500 {
			b.fail()
		} else {
			b.success()
		}
		return
	}
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 500 {
		b.fail()
	} else {
		b.success()
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json; charset=utf-8"
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		g.idem.set(key, fingerprint, owner, resp.StatusCode, contentType, data)
	}
	c.Data(resp.StatusCode, contentType, data)
}

type grpcAskPayload struct {
	RequestID           string `json:"request_id"`
	SessionID           string `json:"session_id"`
	UserID              string `json:"user_id"`
	Query               string `json:"query"`
	Language            string `json:"language"`
	ConversationHistory []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"conversation_history"`
}

func (g *gateway) grpcHeaders(c *gin.Context) metadata.MD {
	values := []string{}
	for _, header := range []string{"authorization", "x-user-id", "x-session-id", "x-user-signature", "x-request-id", "traceparent", "tracestate"} {
		if value := c.GetHeader(header); value != "" {
			values = append(values, header, value)
		}
	}
	return metadata.Pairs(values...)
}

func (g *gateway) grpcCall(ctx context.Context, c *gin.Context, body []byte) (askResponse, error) {
	request, err := g.grpcRequest(c, body)
	if err != nil {
		return askResponse{}, err
	}
	callContext := metadata.NewOutgoingContext(ctx, g.grpcHeaders(c))
	var response askResponse
	err = g.grpcConn.Invoke(callContext, "/mathagent.v1.MathAgent/Ask", request, &response, grpc.ForceCodec(protoCodec{}))
	return response, err
}

func (g *gateway) grpcRequest(c *gin.Context, body []byte) (*askRequest, error) {
	var payload grpcAskPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	if payload.Language == "" {
		payload.Language = "zh"
	}
	userID := c.GetHeader("X-User-ID")
	if userID == "" {
		userID = payload.UserID
	}
	sessionID := c.GetHeader("X-Session-ID")
	if sessionID == "" {
		sessionID = payload.SessionID
	}
	request := &askRequest{
		requestID:   payload.RequestID,
		sessionID:   sessionID,
		userID:      userID,
		query:       payload.Query,
		language:    payload.Language,
		payloadJSON: string(body),
	}
	for _, item := range payload.ConversationHistory {
		request.history = append(request.history, protoMessage{role: item.Role, content: item.Content})
	}
	return request, nil
}

func (g *gateway) grpcStream(ctx context.Context, c *gin.Context, body []byte) (grpc.ClientStream, error) {
	request, err := g.grpcRequest(c, body)
	if err != nil {
		return nil, err
	}
	callContext := metadata.NewOutgoingContext(ctx, g.grpcHeaders(c))
	stream, err := g.grpcConn.NewStream(callContext, &grpc.StreamDesc{StreamName: "AskStream", ServerStreams: true}, "/mathagent.v1.MathAgent/AskStream", grpc.ForceCodec(protoCodec{}))
	if err != nil {
		return nil, err
	}
	if err := stream.SendMsg(request); err != nil {
		return nil, err
	}
	if err := stream.CloseSend(); err != nil {
		return nil, err
	}
	return stream, nil
}

func grpcHTTPStatus(err error) int {
	switch status.Code(err) {
	case codes.InvalidArgument:
		return http.StatusBadRequest
	case codes.Unauthenticated, codes.PermissionDenied:
		return http.StatusUnauthorized
	case codes.ResourceExhausted:
		return http.StatusTooManyRequests
	case codes.DeadlineExceeded:
		return http.StatusGatewayTimeout
	case codes.Unavailable:
		return http.StatusServiceUnavailable
	default:
		return http.StatusBadGateway
	}
}

func (g *gateway) askGRPC(c *gin.Context, body []byte, key, fingerprint, owner string, wantsStream bool) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 45*time.Second)
	defer cancel()
	if !wantsStream {
		response, err := g.grpcCall(ctx, c, body)
		if err != nil {
			g.breakerFor("python-ai").rpcError(err)
			c.JSON(grpcHTTPStatus(err), gin.H{"error": "gRPC 推理服务不可用", "code": status.Code(err).String()})
			return
		}
		g.breakerFor("python-ai").success()
		data := []byte(response.payloadJSON)
		if len(data) == 0 {
			data, _ = json.Marshal(response)
		}
		contentType := "application/json; charset=utf-8"
		g.idem.set(key, fingerprint, owner, http.StatusOK, contentType, data)
		c.Data(http.StatusOK, contentType, data)
		return
	}

	c.Header("Content-Type", "text/event-stream; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")
	c.Status(http.StatusOK)
	flusher, _ := c.Writer.(http.Flusher)
	writeEvent := func(name string, value any) {
		data, _ := json.Marshal(value)
		_, _ = c.Writer.WriteString("event: " + name + "\n")
		_, _ = c.Writer.WriteString("data: " + string(data) + "\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
	stream, err := g.grpcStream(ctx, c, body)
	if err != nil {
		g.breakerFor("python-ai").rpcError(err)
		writeEvent("error", map[string]string{"code": status.Code(err).String()})
		writeEvent("done", map[string]bool{"ok": false})
		return
	}
	seenResult := false
	for {
		var event askEvent
		if err := stream.RecvMsg(&event); err != nil {
			if err == io.EOF {
				if !seenResult {
					g.breakerFor("python-ai").fail()
					writeEvent("error", map[string]string{"code": "INCOMPLETE_STREAM"})
				} else {
					g.breakerFor("python-ai").success()
				}
				writeEvent("done", map[string]bool{"ok": seenResult})
				return
			}
			g.breakerFor("python-ai").rpcError(err)
			writeEvent("error", map[string]string{"code": status.Code(err).String()})
			writeEvent("done", map[string]bool{"ok": false})
			return
		}
		switch event.event {
		case "answer":
			writeEvent("answer", map[string]string{"delta": event.delta})
		case "result":
			if json.Valid([]byte(event.payloadJSON)) {
				seenResult = true
				writeEvent("result", json.RawMessage(event.payloadJSON))
			}
		case "done":
			if !seenResult {
				g.breakerFor("python-ai").fail()
				writeEvent("error", map[string]string{"code": "INCOMPLETE_STREAM"})
			} else {
				g.breakerFor("python-ai").success()
			}
			writeEvent("done", map[string]bool{"ok": seenResult})
			return
		default:
			var payload any
			if event.payloadJSON != "" && json.Unmarshal([]byte(event.payloadJSON), &payload) == nil {
				writeEvent(event.event, payload)
			} else {
				writeEvent(event.event, map[string]string{"delta": event.delta})
			}
		}
	}
}

func (g *gateway) readyHandler(c *gin.Context) {
	if !g.idem.ready() {
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "degraded", "reason": "idempotency_store"})
		return
	}
	if g.useGRPC {
		state := g.grpcConn.GetState()
		if state != connectivity.Ready {
			g.grpcConn.Connect()
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "degraded", "transport": "grpc", "state": state.String()})
			return
		}
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()
		var response readinessResponse
		err := g.grpcConn.Invoke(ctx, "/mathagent.v1.MathAgent/Readiness", &readinessRequest{}, &response, grpc.ForceCodec(protoCodec{}))
		var checks map[string]bool
		_ = json.Unmarshal([]byte(response.checksJSON), &checks)
		if err != nil || response.status != "ready" {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "degraded", "transport": "grpc", "checks": checks})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "ready", "transport": "grpc", "checks": checks})
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, g.upstream+"/ready", nil)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "degraded"})
		return
	}
	resp, err := g.ready.Do(req)
	if err != nil || resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if resp != nil {
			resp.Body.Close()
		}
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "degraded"})
		return
	}
	defer resp.Body.Close()
	c.JSON(http.StatusOK, gin.H{"status": "ready"})
}

type byteReader struct {
	b []byte
	i int
}

func (r *byteReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.i:])
	r.i += n
	return n, nil
}

func maxGatewayInflight() int {
	limit, err := strconv.Atoi(os.Getenv("GATEWAY_MAX_INFLIGHT"))
	if err != nil || limit < 1 || limit > 4096 {
		return 256
	}
	return limit
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	ps, _ := strconv.Atoi(os.Getenv("GATEWAY_RATE_PER_SECOND"))
	if ps <= 0 {
		ps = 80
	}
	burst, _ := strconv.Atoi(os.Getenv("GATEWAY_RATE_BURST"))
	if burst <= 0 {
		burst = 160
	}
	up := os.Getenv("AI_UPSTREAM")
	if up == "" {
		up = "http://app:8000"
	}
	protocol := strings.ToLower(os.Getenv("AI_PROTOCOL"))
	if protocol == "" {
		protocol = "http"
	}
	grpcTarget := os.Getenv("AI_GRPC_TARGET")
	if grpcTarget == "" {
		grpcTarget = "localhost:50051"
	}
	ttlSeconds, _ := strconv.Atoi(os.Getenv("GATEWAY_IDEMPOTENCY_TTL_SECONDS"))
	if ttlSeconds <= 0 {
		ttlSeconds = 3600
	}
	redisURL := os.Getenv("GATEWAY_REDIS_URL")
	g := &gateway{client: &http.Client{Timeout: 46 * time.Second}, ready: &http.Client{Timeout: 3 * time.Second}, upstream: strings.TrimRight(up, "/"), sem: make(chan struct{}, maxGatewayInflight()), idem: newIdemStore(time.Duration(ttlSeconds)*time.Second, redisURL), useGRPC: protocol == "grpc"}
	if g.useGRPC {
		conn, err := grpc.Dial(grpcTarget, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			panic(err)
		}
		g.grpcConn = conn
		defer conn.Close()
	}
	r := newRouter(g, ps, burst)
	port := os.Getenv("GATEWAY_PORT")
	if port == "" {
		port = "8080"
	}
	_ = r.Run(":" + port)
}
