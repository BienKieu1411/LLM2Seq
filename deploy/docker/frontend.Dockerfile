FROM node:22-alpine AS build

WORKDIR /app

ARG VITE_API_BASE=""
ENV VITE_API_BASE=${VITE_API_BASE}

COPY App/frontend/package*.json ./
RUN npm ci

COPY App/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY App/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
