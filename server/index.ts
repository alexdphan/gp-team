// server/index.ts

const express = require("express");
const cors = require("cors");

const app = express();

// Enable CORS for all routes
app.use(cors());

// Your existing route handlers
app.post("/process", (req: any, res: any) => {
  // Handle the request here.
  console.log(req.body);
  res.json({ response: "Hello World" });
});

// Start the server
const PORT = 8000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
