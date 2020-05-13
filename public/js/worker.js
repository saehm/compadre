import { expose, Transfer } from "threads/worker";
import * as d3 from "d3";
import Dexie from "dexie";
import {importDB, exportDB} from "dexie-export-import";
importScripts("./druid.js");
importScripts("./science.js");
importScripts("./reorder.js");

const db = new Dexie("compadre");
db.version(1).stores({
    data: "name, dimensions, labels, names, values",
    subspaces: "++id, dataId, name, columns",
    hddistances: "[subspaceId+metric], distances",
    projections: "++id, subspaceId, name, seed, metric, parameters, status",
    embeddings: "projectionId, projection, distances",
    reorderings: "[left_projectionId+right_projectionId+type], reordering"
});

expose({
    "set_data": set_data,
    "get_subspaces": get_subspaces,
    "convert_data": convert_data,
    "get_projections": get_projections,
    "get_reorderings": get_reorderings,
    "compute_projection": compute_projection,
    "set_status_of_projection": set_status_of_projection,
    "create_subspace": create_subspace,
    "draw": draw,
    "draw_projection": draw_projection,
    "draw_matrix": draw_matrix,
    "draw_comparison": draw_comparison,
    "hover": hover,
    "get_embedding": get_embedding,
    "get_discrepancies": get_discrepancies,
    "reorder_matrix": reorder_matrix,
    "reorder_matrix_within": reorder_matrix_within,
    "delete_db": delete_db,
    "get_db": get_db,
    "load_db": load_db,
})

let _ = {};

async function delete_db() {
    await db.delete();
}

async function get_db() {
    const blob = await exportDB(db);
    return blob;
}

async function load_db(db_blob) {
    await db.delete();
    await Dexie.waitFor(await importDB(db_blob, {"progressCallback": (p => console.log("worker: ", p))}));
}

async function convert_data(data, values) {
    _.data = data
   // _.appearance = appearance

    return new Promise(async (resolve) => {
        const db_data = await db.data
            .where({"name": data.name})
            .toArray();

        if (db_data.length > 0) {
            Object.assign(_, db_data[0]);
            resolve(_);
        } else {
            _.name = data.name
            _.names = get_names(values);
            _.labels = get_labels(values);
            _.dimensions = get_dimensions(values);
            _.values = values.map(row => _.dimensions.map(d => +row[d]));
            
            _.id = await db.data.put({
                name: _.name,
                dimensions: _.dimensions, 
                labels: _.labels, 
                names: _.names, 
                values: _.values,
            });
            resolve(_);
        }
    });
}

async function get_subspaces() {
    return new Promise(async (resolve) => {
        const db_subspaces = await db.subspaces
            .where({
                "dataId": _.name,
                /* "metric": _.data.metric, */
            })
            .toArray();

        if (db_subspaces.length > 0) {
            _.subspaces = db_subspaces;
            resolve(db_subspaces);
        } else {
            /* const N = _.values.length;
            const metric = druid.euclidean;
            const distances = new druid.Matrix(N, N);
            for (let i = 0; i < N; ++i) {
                for (let j = i + 1; j < N; ++j) {
                    const distance = metric(_.values[i], _.values[j])
                    distances.set_entry(i, j, distance);
                    distances.set_entry(j, i, distance);
                }
            } */
            _.subspaces = []
            const subspace = {
                "dataId": _.name,
                "name": "full space", 
                "columns": _.dimensions.map(() => true), 
                //"distances": distances.to2dArray
            }
            
            subspace.id = await db.subspaces.put(subspace);
            _.subspaces.push(subspace);
            resolve(_.subspaces);
        }
    });
}

async function create_subspace(name, cols) {
    const dataId = _.name;
    const subspace = {
        "dataId": dataId,
        "name": name,
        "columns": cols,
    };
    const subspaceId = await db.subspaces.put(subspace);
    subspace.id = subspaceId;
    return subspace;
}

async function get_projections() {
    const subspaceIds = _.subspaces.map(d => d.id);
    return await db.projections
        .where("subspaceId")
        .anyOf(...subspaceIds)
        .toArray();
}

async function get_reorderings() {
    const projections = await get_projections();
    const projectionIds = projections.map(d => d.id);
    const reorderings = await db.reorderings
        .filter(reorder => {
            const id = reorder.left_projectionId
            return projectionIds.findIndex(i => i == id) >= 0;
        })
        //.where(["left_projectionId", "right_projectionId", "type"]).anyOf(projectionIds.map(d => [d, null, null]))
        //.where(0).anyOf(projectionIds)
        .toArray()
    console.log("get_reorderings", reorderings)
    return reorderings;
}

function set_data(O) {
    _ = O;
}

function get_names(data) {
    const has_name = data.columns.findIndex(d => d == "name") >= 0;
    const N = data.length;
    return has_name ? data.map(d => d.name) : druid.linspace(0, N);
}

function get_labels(data) {
    const possible_names = ["label", "class"];
    console.log(data.columns)
    const has_label = data.columns.find(d => possible_names.indexOf(d) >= 0);
    return has_label ? data.map(d => d[has_label]) : new Array(N).fill(0); 
}

function get_dimensions(data) {
    const possible = ["label", "class", "name"];
    return data.columns.filter(d => possible.indexOf(d) < 0);
}

async function get_hddistances(metric, subspaceId, columns) {
    metric = metric ? metric : "euclidean";
    let distances = await db.hddistances.get({
        "subspaceId": subspaceId,
        "metric": metric,
    });

    if (distances == undefined) {
        const values = _.values.map(v => v.filter((_, i) => columns[i]))
        distances = compute_distances(values, druid[metric]);
        await db.hddistances.put({
            "subspaceId": subspaceId,
            "metric": metric,
            "distances": distances,
        });
        return distances;
    } else {
        return distances.distances;
    }   
}

async function compute_projection(projection, subspace) {
    const data = druid.Matrix.from(_.values.map(v => v.filter((_, i) => subspace.columns[i])));
    console.log("compute projection worker", data, projection, subspace)
    const parameters = projection.parameters.filter(d => d.value);
    const seed = projection.seed;
    const metric = projection.metric ? druid[projection.metric] : druid.euclidean;
    const distances = await get_hddistances(projection.metric || "euclidean", subspace.id, subspace.columns);
    let dr = null;
    switch (projection.name) {
        case "tSNE": 
            dr = new druid.TSNE(data, parameters[1].value, parameters[0].value, 2, metric, seed);
            dr.init(druid.Matrix.from(distances));
            console.log(dr)
            projection.projection = dr.transform().to2dArray;
            break;
        case "UMAP":
            dr = new druid.UMAP(data, parameters[1].value, parameters[0].value, 2, metric, seed);
            dr.init();
            projection.projection = dr.transform(350).to2dArray;
            break;
        case "TriMap":
            dr = new druid.TriMap(data, parameters[1].value, parameters[0].value, 2, metric, new druid.Randomizer(seed));
            dr.init();
            projection.projection = dr.transform().to2dArray;
            break;
        case "MDS":
            dr = new druid.MDS(data, 2, metric, seed)
            projection.projection = dr.transform().to2dArray;
            break;
        case "PCA":
            //console.log(data)
            dr = new druid.PCA(data, 2);
            projection.projection = dr.transform().to2dArray;
            break;
        case "ISOMAP":
            dr = new druid.ISOMAP(data, parameters[0].value, 2, metric, seed);
            projection.projection = dr.transform().to2dArray;
            break;
        case "LLE":
            dr = new druid.LLE(data, parameters[0].value, 2, metric, seed);
            projection.projection = dr.transform().to2dArray;
            break;
        case "LTSA":
            dr = new druid.LTSA(data, parameters[0].value, 2, metric, seed);
            projection.projection = dr.transform().to2dArray;
            break;

    }
    /* const N = projection.projection.length;
    projection.distances = new druid.Matrix(N, N);
    for (let i = 0; i < N; ++i) {
        for (let j = i + 1; j < N; ++j) {
            const dist = metric(projection.projection[i], projection.projection[j]);
            projection.distances.set_entry(i, j, dist);
            projection.distances.set_entry(j, i, dist);
        }
    } */
    projection.distances = compute_distances(projection.projection, metric); //projection.distances.to2dArray;
    //"++id, subspaceId, dr_method, seed, metric, parameters, projection, distances",
    projection.id = await db.projections.put({
        "subspaceId": subspace.id,
        "name": projection.name, 
        "seed": seed,
        "metric": projection.metric,
        "parameters": parameters, 
        "status": null,
    });

    await db.embeddings.put({
        "projectionId": projection.id,
        "projection": projection.projection, 
        "distances": projection.distances,
    })

    return projection;
}

function compute_distances(A, metric) {
    const N = A.length;
    const D = new druid.Matrix(N, N);
    for (let i = 0; i < N; ++i) {
        for (let j = i + 1; j < N; ++j) {
            const dist = metric(A[i], A[j]);
            D.set_entry(i, j, dist);
            D.set_entry(j, i, dist);
        }
    }
    return D.to2dArray;
}

async function set_status_of_projection(projectionId, status) {
    await db.projections.update(projectionId, {"status": status});
}

async function get_embedding(projectionId) {
    return await db.embeddings.get({"projectionId": projectionId});
}

async function draw(projection, left_canvas, right_canvas, width, dpr) {
    console.log("worker: draw")
    const embedding = await db.embeddings.get({"projectionId": projection.id})
    draw_projection(embedding.projection, left_canvas, width, dpr)
    draw_matrix(embedding.distances, await get_hddistances(projection.metric, projection.subspaceId), right_canvas, width, dpr)
    return [Transfer(left_canvas), Transfer(right_canvas)];
}

async function get_discrepancies(left_projectionId, right_projectionId) {
    const left_embedding = await db.embeddings.get({"projectionId": left_projectionId});
    const left_projection = await db.projections.get({"id": left_projectionId})
    const left_distances = left_embedding.distances;
    let right_distances;
    if (right_projectionId != "null") {
        const right_embedding = await db.embeddings.get({"projectionId": right_projectionId});
        right_distances = right_embedding.distances;
    } else {
        right_distances = await get_hddistances(left_projection.metric, left_projection.subspaceId);

    }
    return compute_discrepancies(left_distances, right_distances);
}

async function reorder_matrix(left_projectionId, right_projectionId, type) {
    let reordering = await db.reorderings.get({
        "left_projectionId": left_projectionId,
        "right_projectionId": right_projectionId,
        "type": type,
    });

    if (reordering == undefined) {
        const matrix = await get_discrepancies(left_projectionId, right_projectionId);
        const graph = reorder.mat2graph(matrix);
        switch (type) {
            case "olo":
                reordering = reorder.optimal_leaf_order()
                    .distance(druid.euclidean_squared)(matrix);
            break;
            case "bo":
                const barycenter = reorder.barycenter_order(graph);
                const improved = reorder.adjacent_exchange(graph, barycenter[0], barycenter[1])
                reordering = improved[0]
            break;
            case "so":
                reordering = reorder.spectral_order(graph);
            break;
        }
        await db.reorderings.put({
            "left_projectionId": left_projectionId,
            "right_projectionId": right_projectionId,
            "type": type,
            "reordering": reordering,
        });
    }
    return await db.reorderings.get({
        "left_projectionId": left_projectionId,
        "right_projectionId": right_projectionId,
        "type": type,
    });
}

async function reorder_matrix_within(left_projectionId, right_projectionId, type) {
    let reordering = await db.reorderings.get({
        "left_projectionId": left_projectionId,
        "right_projectionId": right_projectionId,
        "type": type,
    });

    if (reordering == undefined) {
        const matrix = await get_discrepancies(left_projectionId, right_projectionId);
        const labels = _.labels;

        const unique_labels = Array.from(new Set(labels));
        const enum_labels = labels.map((d, i) => [d, i]);

        reordering = unique_labels.map(label => {
            const indices = enum_labels
                .filter(d => d[0] == label)
                .map(d => d[1]);
            const N = indices.length;
            const M = (new druid.Matrix(N, N, (i, j) => matrix[indices[i]][indices[j]])).to2dArray;
            const graph = reorder.mat2graph(M);
            let this_reorder;
            switch (type) {
                case "wolo":
                    this_reorder = reorder.optimal_leaf_order()
                        .distance(druid.euclidean_squared)(M);
                break;
                case "wso":
                    this_reorder = reorder.spectral_order(graph);
                break;
            }
            return this_reorder.map(d => indices[d])
        })
        
        const M = (new druid.Matrix(unique_labels.length, unique_labels.length, (i, j) => {
            const row_indices = reordering[i];
            const n = row_indices.length;
            const col_indices = reordering[j];
            const m = col_indices.length;
            return d3.mean(new druid.Matrix(n, m, (k, l) => {
                return matrix[row_indices[k]][col_indices[l]];
            }).to2dArray.flat());
        })).to2dArray
        const graph = reorder.mat2graph(M);

        let between_reorder 
        switch (type) {
            case "wolo":
                between_reorder = reorder.optimal_leaf_order()
                .distance(druid.euclidean_squared)(M)
            break;
            case "wso":
                between_reorder = reorder.spectral_order(graph);
            break;
        }

        reordering = between_reorder.map(ci => reordering[ci]).flat();
        //.flat();
        
        await db.reorderings.put({
            "left_projectionId": left_projectionId,
            "right_projectionId": right_projectionId,
            "type": type,
            "reordering": reordering,
        });
    }
    return await db.reorderings.get({
        "left_projectionId": left_projectionId,
        "right_projectionId": right_projectionId,
        "type": type,
    });
}

function compute_discrepancies(A, B) {
    const a_max = d3.max(A.flat());
    const b_max = d3.max(B.flat());
    const N = A.length;

    const D = new druid.Matrix(N, N, (i, j) => {
        return A[i][j] / a_max - B[i][j] / b_max;
    })

    return D.to2dArray;
}

async function draw_comparison(left_projection, right_projection, left_canvas, mid_canvas, right_canvas, width, dpr) {
    const left_embedding = await db.embeddings.get({"projectionId": left_projection.id});
    const right_embedding = await db.embeddings.get({"projectionId": right_projection.id});
    draw_projection(left_embedding.projection, left_canvas, width, dpr);
    draw_projection(right_embedding.projection, right_canvas, width, dpr);
    draw_matrix(left_embedding.distances, right_embedding.distances, mid_canvas, width, dpr);
}

function draw_matrix(lD, hD, canvas, width, dpr) {
    const color = d3.scaleOrdinal(d3.schemeTableau10);
    const context = canvas.getContext("2d");
    context.scale(dpr, dpr);

    const N = lD.length;
    const D = new druid.Matrix(N, N, (i, j) => lD[i][j] - hD[i][j]);

    const x = d3.scaleLinear()
        .domain([0, N]).range([40, width - 10])
    const w = x(1) - x(0);

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < N; ++i) {
        for (let j = 0; j < N; ++j) {
            const v = D.entry(i, j);
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }

    const c = d3.scaleLinear()
        .domain([min, 0, max])
        .range(["steelblue", "transparent", "tomato"]);

    for (let i = 0; i < N; ++i) {
        context.fillStyle = color(_.labels[i]);
        context.fillRect(x(i), 10, w, 20);
        context.fillRect(10, x(i), 20, w);
        for (let j = 0; j < N; ++j) {
            const v = D.entry(i, j);
            context.fillStyle = c(v);
            context.fillRect(x(i), x(j), w, w);
        }
    }

    context.commit();
}

function draw_projection(projection, canvas, width, dpr) {
    const color = d3.scaleOrdinal(d3.schemeTableau10);
    const context = canvas.getContext("2d");
    context.scale(dpr, dpr)
    const [x, y] = get_xy(projection, width);
    console.log(hover_index)
    projection.forEach((point, i) => {
        const px = x(point[0]);
        const py = y(point[1]);
        context.beginPath();
        context.arc(px, py, 2, 0, Math.PI * 2);
        context.strokeStyle = color(_.labels[i])
        context.strokeStyle = (hover_index == i || !hover_index) ? context.strokeStyle : "lightgrey";
        context.stroke();
    })
    context.commit();
}

function get_xy(projection, width) {
    const x_extent = d3.extent(projection, d => d[0]);
    const y_extent = d3.extent(projection, d => d[1]);

    const x_span = x_extent[1] - x_extent[0];
    const y_span = y_extent[1] - y_extent[0];

    const offset = Math.abs(x_span - y_span) / 2;

    if (x_span > y_span) {
        y_extent[0] -= offset;
        y_extent[1] += offset;
    } else {
        x_extent[0] -= offset;
        x_extent[1] += offset;
    }

    const margin = 10;

    const x = d3.scaleLinear()
        .domain(x_extent)
        .range([margin, width - margin]);

    const y = d3.scaleLinear()
        .domain(y_extent)
        .range([margin, width - margin]);

    return [x, y];
}

async function hover(projection, mouse_x, mouse_y, width, dpr) {
    const embedding = await db.embeddings.get({"projectionId": projection.id});
    let quadtree;
    if (quadtrees.has(projection.id)) {
        console.log("get")
        quadtree = quadtrees.get(projection.id);
    } else {
        console.log("set")
        quadtree = d3.quadtree()
            .addAll(embedding.projection);
        quadtrees.set(projection.id, quadtree);
    }
    const [x, y] = get_xy(embedding.projection, width);
    const found = quadtree.find(x.invert(mouse_x), y.invert(mouse_y));
    const index = embedding.projection.findIndex(point => {
        return point[0] == found[0] && point[1] == found[1];
    })
    hover_index = index;
    return index;
}
let hover_index = null;
const quadtrees = new Map();

