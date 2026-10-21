package main

import (
	"net"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

func TestCapacityErrorsDoNotOpenCircuit(t *testing.T) {
	b := &breaker{}
	for i := 0; i < 10; i++ {
		b.rpcError(status.Error(codes.ResourceExhausted, "busy"))
	}
	if !b.allow() || b.failures.Load() != 0 {
		t.Fatal("admission rejection tripped the upstream circuit")
	}
}

func TestStreamRequiresFinalResult(t *testing.T) {
	for _, final := range []bool{false, true} {
		t.Run(map[bool]string{false: "incomplete", true: "complete"}[final], func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			server := grpc.NewServer(grpc.ForceServerCodec(protoCodec{}))
			server.RegisterService(&grpc.ServiceDesc{
				ServiceName: "mathagent.v1.MathAgent", HandlerType: (*interface{})(nil),
				Streams: []grpc.StreamDesc{{StreamName: "AskStream", ServerStreams: true,
					Handler: func(_ any, stream grpc.ServerStream) error {
						var request askRequest
						if err := stream.RecvMsg(&request); err != nil {
							return err
						}
						if final {
							if err := stream.SendMsg(&askEvent{event: "result", payloadJSON: `{"answer":"x=4"}`}); err != nil {
								return err
							}
						}
						return stream.SendMsg(&askEvent{event: "done", payloadJSON: `{}`})
					}}},
			}, struct{}{})
			go server.Serve(listener)
			defer server.Stop()
			conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			g := testGateway("", "")
			g.grpcConn, g.useGRPC = conn, true
			response := httptest.NewRecorder()
			context, _ := gin.CreateTestContext(response)
			context.Request = httptest.NewRequest("POST", "/ask/stream", nil)
			g.askGRPC(context, []byte(`{"query":"x"}`), "", "", "", true)
			failed := strings.Contains(response.Body.String(), `"ok":false`)
			if failed == final {
				t.Fatalf("unexpected completion: %s", response.Body.String())
			}
		})
	}
}
