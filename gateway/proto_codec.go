package main

// The gateway deliberately keeps the generated Python protobuf contract as
// the source of truth. These small wire-compatible Go values avoid checking a
// protoc binary into the repository while still using a normal gRPC channel.
import (
	"fmt"
)

type protoMessage struct {
	role    string
	content string
}

type protoSource struct {
	chunkID string
	source  string
	chapter string
}

type askRequest struct {
	requestID   string
	sessionID   string
	userID      string
	query       string
	language    string
	history     []protoMessage
	payloadJSON string
}

type askResponse struct {
	traceID     string
	answer      string
	intent      string
	validated   bool
	sources     []protoSource
	payloadJSON string
}

type askEvent struct {
	event       string
	delta       string
	payloadJSON string
}

type readinessRequest struct{}

type readinessResponse struct {
	status     string
	checksJSON string
}

type protoCodec struct{}

func (protoCodec) Name() string { return "proto" }

func appendVarint(dst []byte, value uint64) []byte {
	for value >= 0x80 {
		dst = append(dst, byte(value)|0x80)
		value >>= 7
	}
	return append(dst, byte(value))
}

func appendStringField(dst []byte, field int, value string) []byte {
	if value == "" {
		return dst
	}
	dst = appendVarint(dst, uint64(field<<3|2))
	dst = appendVarint(dst, uint64(len(value)))
	return append(dst, value...)
}

func appendBoolField(dst []byte, field int, value bool) []byte {
	if !value {
		return dst
	}
	dst = appendVarint(dst, uint64(field<<3))
	return append(dst, 1)
}

func appendMessageField(dst []byte, field int, value []byte) []byte {
	dst = appendVarint(dst, uint64(field<<3|2))
	dst = appendVarint(dst, uint64(len(value)))
	return append(dst, value...)
}

func marshalMessage(value protoMessage) []byte {
	data := appendStringField(nil, 1, value.role)
	return appendStringField(data, 2, value.content)
}

func marshalSource(value protoSource) []byte {
	data := appendStringField(nil, 1, value.chunkID)
	data = appendStringField(data, 2, value.source)
	return appendStringField(data, 3, value.chapter)
}

func marshalAskRequest(value *askRequest) []byte {
	data := appendStringField(nil, 1, value.requestID)
	data = appendStringField(data, 2, value.sessionID)
	data = appendStringField(data, 3, value.userID)
	data = appendStringField(data, 4, value.query)
	data = appendStringField(data, 5, value.language)
	for _, item := range value.history {
		data = appendMessageField(data, 6, marshalMessage(item))
	}
	return appendStringField(data, 7, value.payloadJSON)
}

func marshalAskResponse(value *askResponse) []byte {
	data := appendStringField(nil, 1, value.traceID)
	data = appendStringField(data, 2, value.answer)
	data = appendStringField(data, 3, value.intent)
	data = appendBoolField(data, 4, value.validated)
	for _, item := range value.sources {
		data = appendMessageField(data, 5, marshalSource(item))
	}
	return appendStringField(data, 6, value.payloadJSON)
}

func marshalAskEvent(value *askEvent) []byte {
	data := appendStringField(nil, 1, value.event)
	data = appendStringField(data, 2, value.delta)
	return appendStringField(data, 3, value.payloadJSON)
}

func marshalReadinessRequest(value *readinessRequest) []byte { return nil }

func marshalReadinessResponse(value *readinessResponse) []byte {
	data := appendStringField(nil, 1, value.status)
	return appendStringField(data, 2, value.checksJSON)
}

func readVarint(data []byte, offset *int) (uint64, error) {
	var value uint64
	for shift := uint(0); ; shift += 7 {
		if shift >= 64 || *offset >= len(data) {
			return 0, fmt.Errorf("invalid protobuf varint")
		}
		item := data[*offset]
		*offset = *offset + 1
		value |= uint64(item&0x7f) << shift
		if item&0x80 == 0 {
			return value, nil
		}
	}
}

func readBytes(data []byte, offset *int) ([]byte, error) {
	length, err := readVarint(data, offset)
	if err != nil || length > uint64(len(data)-*offset) {
		return nil, fmt.Errorf("invalid protobuf length")
	}
	start := *offset
	*offset += int(length)
	return data[start:*offset], nil
}

func skipField(data []byte, offset *int, wire int) error {
	switch wire {
	case 0:
		_, err := readVarint(data, offset)
		return err
	case 1:
		if len(data)-*offset < 8 {
			return fmt.Errorf("invalid fixed64 field")
		}
		*offset += 8
		return nil
	case 2:
		_, err := readBytes(data, offset)
		return err
	case 5:
		if len(data)-*offset < 4 {
			return fmt.Errorf("invalid fixed32 field")
		}
		*offset += 4
		return nil
	default:
		return fmt.Errorf("unsupported protobuf wire type %d", wire)
	}
}

func unmarshalMessage(data []byte, value *protoMessage) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		if wire != 2 {
			return fmt.Errorf("message field %d has wrong wire type", field)
		}
		switch field {
		case 1:
			value.role = string(raw)
		case 2:
			value.content = string(raw)
		}
		return nil
	})
}

func unmarshalSource(data []byte, value *protoSource) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		if wire != 2 {
			return fmt.Errorf("source field %d has wrong wire type", field)
		}
		switch field {
		case 1:
			value.chunkID = string(raw)
		case 2:
			value.source = string(raw)
		case 3:
			value.chapter = string(raw)
		}
		return nil
	})
}

type fieldHandler func(field, wire int, raw []byte, number uint64) error

func unmarshalFields(data []byte, handler fieldHandler) error {
	for offset := 0; offset < len(data); {
		tag, err := readVarint(data, &offset)
		if err != nil {
			return err
		}
		field, wire := int(tag>>3), int(tag&7)
		if field <= 0 {
			return fmt.Errorf("invalid protobuf field")
		}
		var raw []byte
		var number uint64
		if wire == 2 {
			raw, err = readBytes(data, &offset)
		} else if wire == 0 {
			number, err = readVarint(data, &offset)
		} else {
			err = skipField(data, &offset, wire)
		}
		if err != nil {
			return err
		}
		if err := handler(field, wire, raw, number); err != nil {
			return err
		}
	}
	return nil
}

func unmarshalAskRequest(data []byte, value *askRequest) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		switch field {
		case 1, 2, 3, 4, 5, 7:
			if wire != 2 {
				return fmt.Errorf("request field %d has wrong wire type", field)
			}
			text := string(raw)
			switch field {
			case 1:
				value.requestID = text
			case 2:
				value.sessionID = text
			case 3:
				value.userID = text
			case 4:
				value.query = text
			case 5:
				value.language = text
			case 7:
				value.payloadJSON = text
			}
		case 6:
			if wire != 2 {
				return fmt.Errorf("request history has wrong wire type")
			}
			item := protoMessage{}
			if err := unmarshalMessage(raw, &item); err != nil {
				return err
			}
			value.history = append(value.history, item)
		default:
			_ = number
		}
		return nil
	})
}

func unmarshalAskResponse(data []byte, value *askResponse) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		switch field {
		case 1, 2, 3, 6:
			if wire != 2 {
				return fmt.Errorf("response field %d has wrong wire type", field)
			}
			text := string(raw)
			switch field {
			case 1:
				value.traceID = text
			case 2:
				value.answer = text
			case 3:
				value.intent = text
			case 6:
				value.payloadJSON = text
			}
		case 4:
			if wire != 0 {
				return fmt.Errorf("response validated has wrong wire type")
			}
			value.validated = number != 0
		case 5:
			if wire != 2 {
				return fmt.Errorf("response sources has wrong wire type")
			}
			item := protoSource{}
			if err := unmarshalSource(raw, &item); err != nil {
				return err
			}
			value.sources = append(value.sources, item)
		default:
			_ = number
		}
		return nil
	})
}

func unmarshalAskEvent(data []byte, value *askEvent) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		_ = number
		if field < 1 || field > 3 || wire != 2 {
			return fmt.Errorf("event field %d has wrong wire type", field)
		}
		switch field {
		case 1:
			value.event = string(raw)
		case 2:
			value.delta = string(raw)
		case 3:
			value.payloadJSON = string(raw)
		}
		return nil
	})
}

func unmarshalReadinessRequest(data []byte, value *readinessRequest) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		_ = field
		_ = wire
		_ = raw
		_ = number
		return nil
	})
}

func unmarshalReadinessResponse(data []byte, value *readinessResponse) error {
	return unmarshalFields(data, func(field, wire int, raw []byte, number uint64) error {
		_ = number
		if wire != 2 {
			return fmt.Errorf("readiness field %d has wrong wire type", field)
		}
		switch field {
		case 1:
			value.status = string(raw)
		case 2:
			value.checksJSON = string(raw)
		}
		return nil
	})
}

func (protoCodec) Marshal(value any) ([]byte, error) {
	switch item := value.(type) {
	case *askRequest:
		return marshalAskRequest(item), nil
	case *askResponse:
		return marshalAskResponse(item), nil
	case *askEvent:
		return marshalAskEvent(item), nil
	case *readinessRequest:
		return marshalReadinessRequest(item), nil
	case *readinessResponse:
		return marshalReadinessResponse(item), nil
	default:
		return nil, fmt.Errorf("unsupported gRPC message type %T", value)
	}
}

func (protoCodec) Unmarshal(data []byte, value any) error {
	switch item := value.(type) {
	case *askRequest:
		return unmarshalAskRequest(data, item)
	case *askResponse:
		return unmarshalAskResponse(data, item)
	case *askEvent:
		return unmarshalAskEvent(data, item)
	case *readinessRequest:
		return unmarshalReadinessRequest(data, item)
	case *readinessResponse:
		return unmarshalReadinessResponse(data, item)
	default:
		return fmt.Errorf("unsupported gRPC message type %T", value)
	}
}
