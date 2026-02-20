import { NextRequest } from "next/server";

export const runtime = "edge";

/**
 * SSE proxy: forwards the streaming response from the FastAPI backend
 * directly to the browser without buffering.
 */
export async function POST(req: NextRequest) {
  const body = await req.json();
  const backendUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const upstream = await fetch(`${backendUrl}/api/agent/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok) {
    return new Response(
      JSON.stringify({ error: "Backend request failed", status: upstream.status }),
      { status: upstream.status, headers: { "Content-Type": "application/json" } }
    );
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}
