import { del } from "@vercel/blob";
import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { timingSafeEqual } from "crypto";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

function passwordsMatch(given: string, expected: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  if (a.length !== b.length || a.length === 0) return false;
  return timingSafeEqual(a, b);
}

export async function POST(request: Request): Promise<NextResponse> {
  const body = (await request.json()) as HandleUploadBody;
  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (_pathname, clientPayload) => {
        let password = "";
        try {
          password = JSON.parse(clientPayload || "{}").password || "";
        } catch {
          password = "";
        }
        const expected = process.env.APP_PASSWORD || "";
        if (!passwordsMatch(password, expected)) {
          throw new Error("Unauthorized");
        }
        return {
          allowedContentTypes: [
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/heic",
            "image/heif",
            "image/avif",
            "image/svg+xml",
            "image/x-icon",
            "image/vnd.microsoft.icon",
          ],
          maximumSizeInBytes: 80 * 1024 * 1024,
          addRandomSuffix: true,
        };
      },
      onUploadCompleted: async () => {},
    });
    return NextResponse.json(jsonResponse);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Upload rejected";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

/** Remove temporary uploads once every branding has used them. */
export async function DELETE(request: Request): Promise<NextResponse> {
  let body: { password?: string; urls?: unknown } = {};
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid body" }, { status: 400 });
  }
  if (!passwordsMatch(body.password || "", process.env.APP_PASSWORD || "")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const urls = (Array.isArray(body.urls) ? body.urls : [])
    .filter((u): u is string => typeof u === "string")
    .filter((u) => {
      try {
        const { protocol, hostname } = new URL(u);
        return protocol === "https:" && hostname.endsWith(".blob.vercel-storage.com");
      } catch {
        return false;
      }
    })
    .slice(0, 100);
  if (urls.length) await del(urls);
  return NextResponse.json({ deleted: urls.length });
}
