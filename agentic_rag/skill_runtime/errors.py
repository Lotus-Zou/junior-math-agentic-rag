"""Runtime errors are classified without leaking provider details to students."""


class SkillRuntimeError(RuntimeError):
    retryable = False

    def __init__(self, message: str, *, safe_message: str = "系统暂时无法处理该请求，请稍后重试。"):
        super().__init__(message)
        self.safe_message = safe_message


class RetryableSkillError(SkillRuntimeError):
    retryable = True


class FatalSkillError(SkillRuntimeError):
    pass


class ManifestError(ValueError):
    pass


class PipelineError(ValueError):
    pass

