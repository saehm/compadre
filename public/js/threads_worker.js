import { expose } from "threads/worker";

importScripts("./js/druid.js", "./js/dexie.js");

const db = new Dexie("compadre");
db.version(2).stores({
    projections: "++id, [name+subspace+dr_method], parameters, projection, distances",
    distances: "++id, name, distances",
});

async function compute(name, subspace, dr_method, parameters) {
    let db_result = await db.projections
        .where({
            "name": name,
            "subspace": subspace,
            "dr_method": dr_method,
        }).toArray();

    db_result = db_result
        .find(r => r.parameters
            .map((d, i) => parameters[i].value == d.value)
            .reduce((a, b) => a && b, true)
        )
    let result = null
    if (db_result) {
        result = {
            "type": "finished", 
            "projection": db_result.result,
            "distances": db_result.distances,
            "name": name,
            "subspace": subspace,
            "dr_method": dr_method,
            "parameters": parameters,
        };
    } else {
        const projection = comp_dr(M, parameters, dr_method);

        const N = projection.length;
        const distances = distance_matrix(projection)
        db.projections.add({
            "name": name,
            "subspace": subspace,
            "dr_method": dr_method,
            "parameters": parameters,
            "result": projection,
            "distances": distances,
        });
        result = {
            "type": "finished", 
            "projection": projection,
            "name": name,
            "subspace": subspace,
            "dr_method": dr_method,
            "parameters": parameters,
            "distances": distances,
        }
    }
    return result;
}

expose({
    "send": send,
    "compute": compute,
});


let M = null;
let D = null;

async function send(data, name) {
    const values = data.values;
    //const name = data.name;
    M = druid.Matrix.from(data.values);
    let db_result = await db.distances
        .where({
            "name": name,
        }).toArray();
    if (db_result.length != 0) {
        D = db_result[0].distances;
    } else {
        D = distance_matrix(values);
        await db.distances.add({
            "name": name,
            "distances": D,
        });
    }
    return D;
}

function distance_matrix(A) {
    const N = A.length;
    const DA = new druid.Matrix(N, N);
    const metric = druid.euclidean;
    for (let i = 0; i < N; ++i) {
        DA.set_entry(i, i, 0)
        for (let j = i + 1; j < N; ++j) {
            const d = metric(A[i], A[j])
            DA.set_entry(i, j, d)
            DA.set_entry(j, i, d)
        }
    }
    return DA.to2dArray;
}

function comp_dr(A, parameters, dr_method) {
    let dr;
    let result;
    let iter;
    let next;
    let progress = 0;
    switch (dr_method) {
        case "UMAP":
            dr = new druid.UMAP(A, parameters[0].value, parameters[1].value)
            dr = dr.init();
            result = dr.transform(350);
            return result.to2dArray;
            break;
        case "tSNE":
            dr = new druid.TSNE(A, parameters[0].value, parameters[1].value);
            dr = dr.init();
            result = dr.transform();
            return result.to2dArray;
            break;
        case "TriMap":
            dr = new druid.TriMap(A, parameters[0].value, parameters[1].value);
            dr = dr.init();
            result = dr.transform();
            return result.to2dArray;
            break;
        case "MDS":
            dr = new druid.MDS(A)
            result = dr.transform();
            return result.to2dArray;
            break;
        case "PCA":
            dr = new druid.PCA(A)
            result = dr.transform();
            return result.to2dArray;
            break;
        case "ISOMAP":
            dr = new druid.ISOMAP(A, parameters[0].value)
            result = dr.transform();
            return result.to2dArray;
            break;

        case "LLE":
            dr = new druid.LLE(A, parameters[0].value)
            result = dr.transform();
            return result.to2dArray;
            break;

        case "LTSA":
            dr = new druid.LTSA(A, parameters[0].value)
            result = dr.transform();
            return result.to2dArray;
            break;
        }
}