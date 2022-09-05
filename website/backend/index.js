const express = require('express');
const PORT=5000
const app=express();
const cors = require('cors')
const apiRoutes = require("./routes")

app.use(express.json());
app.use(cors());
app.use("/api",apiRoutes)


app.listen(PORT,() => {
    console.log(`Server is running on port ${PORT}`);
})