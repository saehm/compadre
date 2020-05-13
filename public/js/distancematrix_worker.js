importScripts("./druid.js");

self.addEventListener("message", function(message) {
    const data = message.data;
    const matrix = data.matrix;
    const [N, D] = data.shape;
    const metric = druid[data.metric];
    const A = new druid.Matrix(N, D);
    A._data = matrix;

    const DA = new druid.Matrix(N, N)
    
    for (let i = 0; i < N; ++i) {
        DA.set_entry(i, i, 0)
        for (let j = i + 1; j < N; ++j) {
            const d = metric(A.row(i), A.row(j))
            DA.set_entry(i, j, d)
            DA.set_entry(j, i, d)
        }
    }

    //this.console.log(DA, "sent")
    self.postMessage({
        data: DA._data,
        shape: DA.shape,
    })
});