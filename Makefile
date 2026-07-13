.PHONY: dev dev-build prod prod-build clean

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

clean:
	docker compose down -v
