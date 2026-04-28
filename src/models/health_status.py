from pydantic import BaseModel, computed_field


class HealthStatus(BaseModel):
    reachable: bool

    @computed_field
    @property
    def status(self) -> str:
        return "ok" if self.reachable else "degraded"
