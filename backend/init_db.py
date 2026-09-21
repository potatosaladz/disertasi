from database import engine
from models import Agent


def initialize_database() -> None:
    Agent.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
