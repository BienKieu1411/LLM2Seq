# LLM2Seq Frontend

React/Vite frontend for the LLM2Seq summarization demo.

The app provides:

- source text input with a WikiLingua sample loader;
- decoding mode switch between autoregressive and verified MTP;
- maximum-token control;
- generated summary display;
- latency, token count, throughput, and MTP runtime statistics.

## Local Development

Install dependencies:

```bash
cd App/frontend
npm install
```

Run the frontend:

```bash
npm run dev
```

By default, the frontend calls:

```text
http://localhost:8000
```

To point it to a different backend:

```bash
VITE_API_BASE=http://localhost:9000 npm run dev
```

## Production Build

```bash
npm run build
```

The production Docker image builds the frontend with an empty API base:

```text
VITE_API_BASE=
```

The app then calls same-origin `/api/*`, and Nginx proxies those requests to the backend container.

## UI Theme

The interface uses a light neo-brutalist academic-tool style: cream background,
white panels, dark borders, hard shadows, and flat accent colors.

The detailed recreation spec is:

```text
App/frontend/UI_THEME_SPEC.md
```
