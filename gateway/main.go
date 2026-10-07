package main

// C 端接入层：收敛 HTTP/SSE 流量，执行限流、并发保护、幂等与上游熔断。
// AI 推理仍由 Python 服务负责，网关不承载模型状态。
import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

type breaker struct { failures atomic.Int64; opened atomic.Int64 }
func (b *breaker) allow() bool { return time.Now().UnixNano() > b.opened.Load() }
func (b *breaker) success() { b.failures.Store(0) }
func (b *breaker) fail() { if b.failures.Add(1) >= 5 { b.opened.Store(time.Now().Add(10*time.Second).UnixNano()) } }

type gateway struct {
	client *http.Client
	upstream string
	sem chan struct{}
	breakers sync.Map
	idem *sync.Map
}

func (g *gateway) breakerFor(key string) *breaker { v, _ := g.breakers.LoadOrStore(key, &breaker{}); return v.(*breaker) }
func requestKey(c *gin.Context) string { if v := c.GetHeader("Idempotency-Key"); v != "" { return v }; h := sha256.Sum256([]byte(c.Request.Method+"|"+c.Request.URL.Path+"|"+c.ClientIP())); return hex.EncodeToString(h[:]) }

func (g *gateway) ask(c *gin.Context) {
	select { case g.sem <- struct{}{}: defer func(){<-g.sem}(); default: c.JSON(http.StatusTooManyRequests, gin.H{"error":"请求排队已满，请稍后重试","code":"QUEUE_FULL"}); return }
	b := g.breakerFor("python-ai"); if !b.allow() { c.JSON(http.StatusServiceUnavailable, gin.H{"error":"AI 服务正在恢复，请稍后重试","code":"UPSTREAM_OPEN"}); return }
	key := requestKey(c); if value, ok := g.idem.Load(key); ok { c.Data(http.StatusOK, "application/json; charset=utf-8", value.([]byte)); return }
	body, err := io.ReadAll(io.LimitReader(c.Request.Body, 1<<20)); if err != nil { c.JSON(http.StatusBadRequest, gin.H{"error":"请求体读取失败"}); return }
	ctx, cancel := context.WithTimeout(c.Request.Context(), 45*time.Second); defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, g.upstream+"/ask", io.NopCloser(bytesReader(body)))
	req.Header.Set("Content-Type", "application/json"); req.Header.Set("X-Request-ID", c.GetHeader("X-Request-ID"))
	resp, err := g.client.Do(req); if err != nil { b.fail(); c.JSON(http.StatusGatewayTimeout, gin.H{"error":"推理服务暂时不可用","code":"UPSTREAM_TIMEOUT"}); return }
	defer resp.Body.Close(); data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 500 { b.fail() } else { b.success() }
	if resp.StatusCode < 500 { g.idem.Store(key, data) }
	c.Data(resp.StatusCode, "application/json; charset=utf-8", data)
}

// bytesReader 避免为每次请求暴露可变 buffer。
type byteReader struct { b []byte; i int }; func (r *byteReader) Read(p []byte)(int,error){ if r.i>=len(r.b){return 0,io.EOF}; n:=copy(p,r.b[r.i:]); r.i+=n; return n,nil }
func bytesReader(b []byte) io.Reader { return &byteReader{b:b} }

func main() {
	gin.SetMode(gin.ReleaseMode); r:=gin.New(); r.Use(gin.Recovery())
	perSecond,_:=strconv.Atoi(os.Getenv("GATEWAY_RATE_PER_SECOND")); if perSecond<=0 { perSecond=80 }
	burst,_:=strconv.Atoi(os.Getenv("GATEWAY_RATE_BURST")); if burst<=0 { burst=160 }
	limiter:=rate.NewLimiter(rate.Limit(perSecond), burst); g:=&gateway{client:&http.Client{Timeout:46*time.Second}, upstream:os.Getenv("AI_UPSTREAM")}; if g.upstream=="" {g.upstream="http://app:8000"}; g.sem=make(chan struct{}, 256); g.idem=&sync.Map{}
	r.Use(func(c *gin.Context){ if !limiter.Allow(){c.JSON(429,gin.H{"error":"请求过于频繁","code":"RATE_LIMITED"}); c.Abort(); return}; c.Next() })
	r.GET("/health",func(c *gin.Context){c.JSON(200,gin.H{"status":"ok","service":"gateway"})}); r.GET("/ready",func(c *gin.Context){c.JSON(200,gin.H{"status":"ready","upstream":g.upstream})}); r.POST("/ask",g.ask)
	port:=os.Getenv("GATEWAY_PORT"); if port=="" {port="8080"}; _=r.Run(":"+port)
}
