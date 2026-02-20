import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow streaming responses from the backend
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Cache-Control", value: "no-cache, no-transform" },
        ],
      },
    ];
  },
};

export default nextConfig;
