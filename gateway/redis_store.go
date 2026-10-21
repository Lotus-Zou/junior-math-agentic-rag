package main

// Minimal Redis RESP client used for cross-replica idempotency state.
// Keeping this client local avoids adding a large gateway dependency for five commands.
import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type redisClient struct {
	addr     string
	password string
	db       string
	pool     chan net.Conn
	dialer   net.Dialer
}

func newRedisClient(raw string, size int) (*redisClient, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" {
		return nil, fmt.Errorf("invalid redis url")
	}
	addr := parsed.Host
	if !strings.Contains(addr, ":") {
		addr += ":6379"
	}
	password := ""
	if parsed.User != nil {
		password, _ = parsed.User.Password()
	}
	db := strings.TrimPrefix(parsed.Path, "/")
	if db == "" {
		db = "0"
	}
	client := &redisClient{addr: addr, password: password, db: db, pool: make(chan net.Conn, max(1, size))}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := client.dial(ctx)
	if err != nil {
		return nil, err
	}
	client.put(conn)
	return client, nil
}

func max(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func (r *redisClient) dial(ctx context.Context) (net.Conn, error) {
	conn, err := r.dialer.DialContext(ctx, "tcp", r.addr)
	if err != nil {
		return nil, err
	}
	if r.password != "" {
		if _, err = r.command(ctx, conn, "AUTH", r.password); err != nil {
			conn.Close()
			return nil, err
		}
	}
	if r.db != "0" {
		if _, err = r.command(ctx, conn, "SELECT", r.db); err != nil {
			conn.Close()
			return nil, err
		}
	}
	return conn, nil
}

func (r *redisClient) get(ctx context.Context) (net.Conn, error) {
	select {
	case conn := <-r.pool:
		return conn, nil
	default:
		return r.dial(ctx)
	}
}

func (r *redisClient) put(conn net.Conn) {
	if conn == nil {
		return
	}
	select {
	case r.pool <- conn:
	default:
		_ = conn.Close()
	}
}

func encodeRESP(args ...string) []byte {
	var b strings.Builder
	fmt.Fprintf(&b, "*%d\r\n", len(args))
	for _, arg := range args {
		fmt.Fprintf(&b, "$%d\r\n%s\r\n", len(arg), arg)
	}
	return []byte(b.String())
}

func (r *redisClient) command(ctx context.Context, conn net.Conn, args ...string) (any, error) {
	deadline, ok := ctx.Deadline()
	if ok {
		_ = conn.SetDeadline(deadline)
	}
	if _, err := conn.Write(encodeRESP(args...)); err != nil {
		return nil, err
	}
	return readRESP(bufio.NewReader(conn))
}

func readRESP(reader *bufio.Reader) (any, error) {
	kind, err := reader.ReadByte()
	if err != nil {
		return nil, err
	}
	line, err := reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
	switch kind {
	case '+':
		return line, nil
	case '-':
		return nil, fmt.Errorf("redis: %s", line)
	case ':':
		return strconv.ParseInt(line, 10, 64)
	case '$':
		n, err := strconv.Atoi(line)
		if err != nil {
			return nil, err
		}
		if n < 0 {
			return nil, nil
		}
		value := make([]byte, n+2)
		if _, err := io.ReadFull(reader, value); err != nil {
			return nil, err
		}
		return string(value[:n]), nil
	case '*':
		n, err := strconv.Atoi(line)
		if err != nil {
			return nil, err
		}
		items := make([]any, n)
		for i := range items {
			items[i], err = readRESP(reader)
			if err != nil {
				return nil, err
			}
		}
		return items, nil
	default:
		return nil, fmt.Errorf("unknown redis response %q", kind)
	}
}

func (r *redisClient) do(ctx context.Context, args ...string) (any, error) {
	conn, err := r.get(ctx)
	if err != nil {
		return nil, err
	}
	result, err := r.command(ctx, conn, args...)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	r.put(conn)
	return result, nil
}
