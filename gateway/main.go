package main

// C 端接入层：限流、并发保护、幂等、熔断与 HTTP/SSE 反向代理。
import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type breaker struct {
	failures atomic.Int64
	opened   atomic.Int64
}

func (b *breaker) allow() bool { return time.Now().UnixNano() > b.opened.Load() }
func (b *breaker) success()    { b.failures.Store(0) }
func (b *breaker) fail() {
	if b.failures.Add(1) >= 5 {
		b.opened.Store(time.Now().Add(10 * time.Second).UnixNano())
	}
}

type gateway struct {
	client   *http.Client
	upstream string
	sem      chan struct{}
	breakers sync.Map
	idem     sync.Map
}

func (g *gateway) breakerFor(key string) *breaker {
	v, _ := g.breakers.LoadOrStore(key, &breaker{})
	return v.(*breaker)
}
func requestKey(c *gin.Context) string {
	if v := c.GetHeader("Idempotency-Key"); v != "" {
		return v
	}
	h := sha256.Sum256([]byte(c.Request.Method + "|" + c.Request.URL.Path + "|" + c.ClientIP()))
	return hex.EncodeToString(h[:])
}
func (g *gateway) ask(c *gin.Context) {
	select {
	case g.sem <- struct{}{}:
		defer func() { <-g.sem }()
	default:
		c.JSON(http.StatusTooManyRequests, gin.H{"error": "请求排队已满", "code": "QUEUE_FULL"})
		return
	}
	b := g.breakerFor("python-ai")
	if !b.allow() {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI 服务熔断恢复中", "code": "UPSTREAM_OPEN"})
		return
	}
	key := requestKey(c)
	body, err := io.ReadAll(io.LimitReader(c.Request.Body, 1<<20))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求体读取失败"})
		return
	}
	wantsStream := c.GetHeader("Accept") == "text/event-stream" || c.Request.URL.Path == "/ask/stream"
	if !wantsStream {
		if value, ok := g.idem.Load(key); ok {
			c.Data(http.StatusOK, "application/json; charset=utf-8", value.([]byte))
			return
		}
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 45*time.Second)
	defer cancel()
	path := c.Request.URL.Path
	if path == "" {
		path = "/ask"
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, g.upstream+path, &byteReader{b: body})
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", c.GetHeader("Accept"))
	req.Header.Set("X-Request-ID", c.GetHeader("X-Request-ID"))
	resp, err := g.client.Do(req)
	if err != nil {
		b.fail()
		c.JSON(http.StatusGatewayTimeout, gin.H{"error": "推理服务超时", "code": "UPSTREAM_TIMEOUT"})
		return
	}
	defer resp.Body.Close()
	if wantsStream || resp.Header.Get("Content-Type") == "text/event-stream" {
		c.Header("Content-Type", "text/event-stream; charset=utf-8")
		c.Header("Cache-Control", "no-cache")
		c.Header("X-Accel-Buffering", "no")
		c.Status(resp.StatusCode)
		f, _ := c.Writer.(http.Flusher)
		buf := make([]byte, 32768)
		for {
			n, e := resp.Body.Read(buf)
			if n > 0 {
				_, _ = c.Writer.Write(buf[:n])
				if f != nil {
					f.Flush()
				}
			}
			if e != nil {
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
	if resp.StatusCode < 500 {
		g.idem.Store(key, data)
	}
	c.Data(resp.StatusCode, "application/json; charset=utf-8", data)
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
func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	ps, _ := strconv.Atoi(os.Getenv("GATEWAY_RATE_PER_SECOND"))
	if ps <= 0 {
		ps = 80
	}
	burst, _ := strconv.Atoi(os.Getenv("GATEWAY_RATE_BURST"))
	if burst <= 0 {
		burst = 160
	}
	limiter := rate.NewLimiter(rate.Limit(ps), burst)
	up := os.Getenv("AI_UPSTREAM")
	if up == "" {
		up = "http://app:8000"
	}
	g := &gateway{client: &http.Client{Timeout: 46 * time.Second}, upstream: up, sem: make(chan struct{}, 256)}
	r.Use(func(c *gin.Context) {
		if !limiter.Allow() {
			c.JSON(429, gin.H{"error": "请求过于频繁", "code": "RATE_LIMITED"})
			c.Abort()
			return
		}
		c.Next()
	})
	r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok", "service": "gateway"}) })
	r.GET("/ready", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ready", "upstream": g.upstream}) })
	r.POST("/ask", g.ask)
	r.POST("/ask/stream", g.ask)
	port := os.Getenv("GATEWAY_PORT")
	if port == "" {
		port = "8080"
	}
	_ = r.Run(":" + port)
}
