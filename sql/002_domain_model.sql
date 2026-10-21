-- Apply with tools/migrate_mysql.py against an existing MySQL database.
-- New Docker databases get the ownership-scoped schema from 001_schema.sql.
-- Existing databases use the metadata-aware Python migration; schema.sql already
-- includes session_key for new MySQL 8 installations.
CREATE INDEX idx_mistake_user_session ON mistake_questions(user_id, session_key, created_at);
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    session_key VARCHAR(128) NOT NULL,
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    trace_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_message_session(user_id, session_key, created_at)
);
