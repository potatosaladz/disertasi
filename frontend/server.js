const http = require("http");

const server = http.createServer((_request, response) => {
  response.writeHead(200, { "Content-Type": "application/json" });
  response.end(JSON.stringify({ status: "SHCR Frontend Placeholder" }));
});

server.listen(3000, "0.0.0.0");
