from app.database import Base, engine
from app.models import Project, Paper


Base.metadata.create_all(engine)

print("Database tables created.")