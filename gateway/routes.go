package main

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

const browserCookie = "math_agent_session"

func randomHex(bytes int) string {
	value := make([]byte, bytes)
	if _, err := rand.Read(value); err != nil {
		panic(err)
	}
	return hex.EncodeToString(value)
}

func signBrowser(value string) string {
	signer := hmac.New(sha256.New, []byte(os.Getenv("AUTH_SHARED_SECRET")))
	_, _ = signer.Write([]byte("browser-session|" + value))
	return hex.EncodeToString(signer.Sum(nil))
}

func browserUser(c *gin.Context) string {
	value, err := c.Cookie(browserCookie)
	if err != nil || os.Getenv("AUTH_SHARED_SECRET") == "" {
		return ""
	}
	parts := strings.Split(value, ".")
	if len(parts) != 3 {
		return ""
	}
	expires, err := strconv.ParseInt(parts[1], 10, 64)
	if err != nil || expires <= time.Now().Unix() || !hmac.Equal([]byte(parts[2]), []byte(signBrowser(parts[0]+"."+parts[1]))) {
		return ""
	}
	return parts[0]
}

func browserSession(c *gin.Context) {
	if !strings.EqualFold(os.Getenv("AUTH_BROWSER_SESSIONS"), "true") || os.Getenv("AUTH_SHARED_SECRET") == "" {
		c.JSON(http.StatusNotFound, gin.H{"code": "BROWSER_SESSIONS_DISABLED"})
		return
	}
	user := browserUser(c)
	if user == "" {
		user = "student-" + randomHex(16)
		value := user + "." + strconv.FormatInt(time.Now().Add(24*time.Hour).Unix(), 10)
		http.SetCookie(c.Writer, &http.Cookie{
			Name: browserCookie, Value: value + "." + signBrowser(value), Path: "/",
			MaxAge: 86400, HttpOnly: true, SameSite: http.SameSiteStrictMode,
			Secure: strings.EqualFold(os.Getenv("AUTH_COOKIE_SECURE"), "true"),
		})
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(http.StatusOK, gin.H{"user_id": user})
}

func browserHeaders(c *gin.Context) {
	if !strings.EqualFold(os.Getenv("AUTH_BROWSER_SESSIONS"), "true") || c.GetHeader("X-User-Signature") != "" {
		return
	}
	user := browserUser(c)
	if user == "" {
		return
	}
	if supplied := c.GetHeader("X-User-ID"); supplied != "" && supplied != user {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"code": "IDENTITY_MISMATCH"})
		return
	}
	c.Request.Header.Set("X-User-ID", user)
	if c.GetHeader("X-Session-ID") == "" {
		c.Request.Header.Set("X-Session-ID", "browser-session")
	}
	signer := hmac.New(sha256.New, []byte(os.Getenv("AUTH_SHARED_SECRET")))
	_, _ = signer.Write([]byte(user))
	c.Request.Header.Set("X-User-Signature", hex.EncodeToString(signer.Sum(nil)))
}

type clientLimit struct {
	limiter *rate.Limiter
	seen    time.Time
}

type routeLimits struct {
	mu       sync.Mutex
	items    map[string]*clientLimit
	ps       int
	burst    int
	overflow *rate.Limiter
}

func (limits *routeLimits) allow(key string) bool {
	limits.mu.Lock()
	defer limits.mu.Unlock()
	item := limits.items[key]
	if item == nil {
		if len(limits.items) >= 10000 {
			for existing, entry := range limits.items {
				if time.Since(entry.seen) > 5*time.Minute {
					delete(limits.items, existing)
				}
			}
			if len(limits.items) >= 10000 {
				return limits.overflow.Allow()
			}
		}
		item = &clientLimit{limiter: rate.NewLimiter(rate.Limit(limits.ps), limits.burst)}
		limits.items[key] = item
	}
	item.seen = time.Now()
	return item.limiter.Allow()
}

func validTraceparent(value string) bool {
	if len(value) != 55 || value[:3] != "00-" || value[35] != '-' || value[52] != '-' {
		return false
	}
	_, traceErr := hex.DecodeString(value[3:35])
	_, spanErr := hex.DecodeString(value[36:52])
	_, flagsErr := hex.DecodeString(value[53:55])
	return traceErr == nil && spanErr == nil && flagsErr == nil && value[3:35] != strings.Repeat("0", 32) && value[36:52] != strings.Repeat("0", 16)
}

func newRouter(g *gateway, ps, burst int) *gin.Engine {
	router := gin.New()
	router.Use(gin.Recovery())
	_ = router.SetTrustedProxies(nil)
	limits := &routeLimits{items: make(map[string]*clientLimit), ps: ps, burst: burst, overflow: rate.NewLimiter(rate.Limit(ps), burst)}
	var metricsMu sync.Mutex
	counts := map[string]int64{}
	router.Use(func(c *gin.Context) {
		started := time.Now()
		incoming := c.GetHeader("traceparent")
		traceID, parentID := randomHex(16), ""
		if validTraceparent(incoming) {
			traceID, parentID = incoming[3:35], incoming[36:52]
		}
		spanID := randomHex(8)
		c.Request.Header.Set("traceparent", "00-"+traceID+"-"+spanID+"-01")
		if c.GetHeader("X-Request-ID") == "" {
			c.Request.Header.Set("X-Request-ID", traceID)
		}
		c.Header("X-Request-ID", c.GetHeader("X-Request-ID"))
		browserHeaders(c)
		if !c.IsAborted() && (strings.HasPrefix(c.Request.URL.Path, "/ask") || c.Request.URL.Path == "/tasks" || c.Request.URL.Path == "/attachments/parse") {
			identity := c.GetHeader("X-User-ID")
			if identity == "" {
				identity = c.ClientIP()
			}
			if !limits.allow(identity + "|" + c.FullPath()) {
				c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "请求过于频繁", "code": "RATE_LIMITED"})
			}
		}
		c.Next()
		path := c.FullPath()
		if path == "" {
			path = "unmatched"
		}
		key := fmt.Sprintf("route=%q,status=%q", path, strconv.Itoa(c.Writer.Status()))
		metricsMu.Lock()
		counts[key]++
		metricsMu.Unlock()
		if path != "/health" && path != "/ready" && path != "/metrics" {
			record, _ := json.Marshal(map[string]any{
				"event": "gateway_request", "trace_id": traceID, "span_id": spanID, "parent_span_id": parentID,
				"request_id": c.GetHeader("X-Request-ID"), "method": c.Request.Method, "route": path,
				"status": c.Writer.Status(), "latency_ms": float64(time.Since(started).Microseconds()) / 1000,
			})
			log.Print(string(record))
		}
	})
	target, err := url.Parse(g.upstream)
	if err != nil {
		panic(err)
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Transport = &http.Transport{MaxIdleConns: 128, MaxIdleConnsPerHost: 32, ResponseHeaderTimeout: 80 * time.Second}
	proxy.ErrorHandler = func(writer http.ResponseWriter, _ *http.Request, _ error) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusServiceUnavailable)
		_, _ = writer.Write([]byte(`{"code":"UPSTREAM_UNAVAILABLE"}`))
	}
	public := func(c *gin.Context) { proxy.ServeHTTP(c.Writer, c.Request) }
	protected := func(c *gin.Context) {
		if g.validateIdentity(c) {
			proxy.ServeHTTP(c.Writer, c.Request)
		}
	}
	router.GET("/health", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok", "service": "gateway"}) })
	router.GET("/ready", g.readyHandler)
	router.GET("/metrics", func(c *gin.Context) {
		metricsMu.Lock()
		defer metricsMu.Unlock()
		var body strings.Builder
		body.WriteString("# TYPE math_gateway_requests_total counter\n")
		for labels, count := range counts {
			fmt.Fprintf(&body, "math_gateway_requests_total{%s} %d\n", labels, count)
		}
		c.Data(http.StatusOK, "text/plain; version=0.0.4", []byte(body.String()))
	})
	router.GET("/auth/session", browserSession)
	router.POST("/ask", g.ask)
	router.POST("/ask/stream", g.ask)
	for _, path := range []string{"/", "/favicon.ico", "/static/*path", "/runtime"} {
		router.GET(path, public)
	}
	for _, path := range []string{"/mistakes", "/reports/learning", "/memory", "/tasks/:task_id"} {
		router.GET(path, protected)
	}
	for _, path := range []string{"/feedback", "/attachments/parse", "/tasks"} {
		router.POST(path, protected)
	}
	router.DELETE("/memory/:memory_id", protected)
	return router
}
