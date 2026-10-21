package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func testGateway(upstream string, redisURL string) *gateway {
	return &gateway{client: &http.Client{Timeout: time.Second}, ready: &http.Client{Timeout: time.Second}, upstream: upstream,
		sem: make(chan struct{}, 4), idem: newIdemStore(time.Minute, redisURL)}
}

func TestGatewayInflightLimitIsConfigurableAndBounded(t *testing.T) {
	for _, test := range []struct {
		value string
		want  int
	}{{"", 256}, {"invalid", 256}, {"0", 256}, {"-1", 256}, {"4097", 256}, {"512", 512}, {"1", 1}} {
		t.Run(test.value, func(t *testing.T) {
			t.Setenv("GATEWAY_MAX_INFLIGHT", test.value)
			if got := maxGatewayInflight(); got != test.want {
				t.Fatalf("got %d, want %d", got, test.want)
			}
		})
	}
}

func TestCompletePublicRoutes(t *testing.T) {
	t.Setenv("AUTH_SHARED_SECRET", "")
	t.Setenv("REQUIRE_USER_IDENTITY", "false")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()
	router := newRouter(testGateway(upstream.URL, ""), 1000, 1000)
	for _, path := range []string{"/", "/runtime", "/static/app.js", "/mistakes", "/reports/learning", "/tasks/example"} {
		w := httptest.NewRecorder()
		ctx, cancel := context.WithCancel(context.Background())
		router.ServeHTTP(w, httptest.NewRequest("GET", path, nil).WithContext(ctx))
		cancel()
		if w.Code != 200 {
			t.Fatalf("%s returned %d", path, w.Code)
		}
	}
	for _, path := range []string{"/feedback", "/attachments/parse", "/tasks"} {
		w := httptest.NewRecorder()
		ctx, cancel := context.WithCancel(context.Background())
		router.ServeHTTP(w, httptest.NewRequest("POST", path, strings.NewReader("{}")).WithContext(ctx))
		cancel()
		if w.Code != 200 {
			t.Fatalf("%s returned %d", path, w.Code)
		}
	}
}

func TestBusyClientDoesNotRateLimitHealthOrOtherUsers(t *testing.T) {
	t.Setenv("AUTH_SHARED_SECRET", "")
	t.Setenv("REQUIRE_USER_IDENTITY", "false")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(`{"answer":"ok"}`)) }))
	defer upstream.Close()
	router := newRouter(testGateway(upstream.URL, ""), 1, 1)
	request := func(user string) int {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/ask", strings.NewReader(`{"query":"x"}`))
		r.Header.Set("X-User-ID", user)
		router.ServeHTTP(w, r)
		return w.Code
	}
	if request("one") != 200 || request("one") != 429 || request("two") != 200 {
		t.Fatal("per-user limits were not independent")
	}
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest("GET", "/health", nil))
	if w.Code != 200 {
		t.Fatal("health was rate limited")
	}
}

func TestRedisConcurrentIdempotency(t *testing.T) {
	redisURL := os.Getenv("GATEWAY_TEST_REDIS_URL")
	if redisURL == "" {
		t.Skip("set GATEWAY_TEST_REDIS_URL to an isolated Redis database")
	}
	t.Setenv("AUTH_SHARED_SECRET", "")
	t.Setenv("REQUIRE_USER_IDENTITY", "false")
	entered, release := make(chan struct{}), make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		<-release
		_, _ = w.Write([]byte(`{"answer":"ok"}`))
	}))
	defer upstream.Close()
	router := newRouter(testGateway(upstream.URL, redisURL), 1000, 1000)
	key := randomHex(16)
	request := func() *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/ask", bytes.NewBufferString(`{"query":"x"}`))
		r.Header.Set("Idempotency-Key", key)
		router.ServeHTTP(w, r)
		return w
	}
	finished := make(chan *httptest.ResponseRecorder, 1)
	go func() { finished <- request() }()
	<-entered
	second := request()
	close(release)
	first := <-finished
	if second.Code != http.StatusConflict || !strings.Contains(second.Body.String(), "IDEMPOTENCY_IN_PROGRESS") {
		t.Fatalf("inflight duplicate returned %d %q", second.Code, second.Body.String())
	}
	third := request()
	if first.Code != 200 || third.Code != 200 || third.Body.String() != first.Body.String() {
		t.Fatal("completed replay did not preserve the response")
	}
}

func TestBrowserSessionUsesServerIssuedIdentity(t *testing.T) {
	t.Setenv("AUTH_SHARED_SECRET", "test-secret")
	t.Setenv("REQUIRE_USER_IDENTITY", "true")
	t.Setenv("AUTH_BROWSER_SESSIONS", "true")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, `{"answer":"ok"}`) }))
	defer upstream.Close()
	router := newRouter(testGateway(upstream.URL, ""), 1000, 1000)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest("GET", "/auth/session", nil))
	var identity map[string]string
	_ = json.Unmarshal(w.Body.Bytes(), &identity)
	if !strings.HasPrefix(identity["user_id"], "student-") || len(w.Result().Cookies()) != 1 {
		t.Fatal("missing server-issued browser session")
	}
	r := httptest.NewRequest("POST", "/ask", strings.NewReader(`{"query":"x"}`))
	r.AddCookie(w.Result().Cookies()[0])
	r.Header.Set("X-User-ID", identity["user_id"])
	r.Header.Set("X-Session-ID", "session")
	w = httptest.NewRecorder()
	router.ServeHTTP(w, r)
	if w.Code != 200 {
		t.Fatalf("browser request failed: %d %s", w.Code, w.Body)
	}
}

func TestBreakerAllowsOnlyOneRecoveryProbe(t *testing.T) {
	b := &breaker{}
	for i := 0; i < 5; i++ {
		b.fail()
	}
	if b.allow() {
		t.Fatal("open breaker allowed a request")
	}
	b.opened.Store(time.Now().Add(-time.Second).UnixNano())
	if !b.allow() || b.allow() {
		t.Fatal("half-open breaker allowed more than one probe")
	}
	b.success()
	if !b.allow() {
		t.Fatal("successful probe did not close breaker")
	}
}

func init() { gin.SetMode(gin.TestMode) }
