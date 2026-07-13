.PHONY: dev dev-build dev-logs prod prod-build prod-down clean test

dev:
	docker compose up -d

dev-build:
	docker compose up -d --build

dev-logs:
	docker compose logs -f

prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-build:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

prod-down:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down

test:
	pytest -c tests/pytest.ini tests/ -v

clean:
	docker compose down -v
