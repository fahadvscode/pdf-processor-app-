/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // pdf.js optionally requires the Node "canvas" package; the browser build never uses it.
    config.resolve.alias.canvas = false;
    return config;
  },
};
module.exports = nextConfig;
