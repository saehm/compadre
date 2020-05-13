import Vue from 'vue';
import Vuex from 'vuex';
import * as d3 from "d3";
import download from "downloadjs";
import { spawn, Worker, Pool, Transfer } from "threads";

import projectionmethods from "./modules/projectionmethods";
import datasets from "./modules/datasets";

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    projection_methods: projectionmethods,
    datasets: datasets,
  },
  state: {
    data_name: "VISPUBDATA",
    points: null,
    labels: null,
    label_color: null,
    dimensions: null,
    values: null,
    subspaces: [],
    projections: [],
    threads: [],
    pool: null,
    draw_pool: null,
    data: {
      "name": "VISPUBDATA",
      "path": "data/vispubdata_authors_keywords_refs.csv",
      "description": "Extracted from vispubdata.org. 578 authors, three subspaces. Dimensions 0-611: coauthor-network, 611-1100: keywords, 1100-1480: citation-network.",
      "appearance": {
        "scale": 1,            
        "pointtype": "point",
        "colorscale": "categorical",
      },
      "sparse": true,
    },
    appearance: {
      "pointtype": "point",
      "colorscale": "categorical",
    },
    comparisons:[],
    workers: [],
    hover: null,
    reorderings: [],
    selection: [],
    images: null,
    loading: null,
    first_projection: null,
    tooltip: null,
  },
  mutations: {
    set_images(state, images) {
      state.images = images;
    },
    set_tooltip(state, tooltip) {
      state.tooltip = tooltip;
    },
    set_hover_index(state, index) {
      state.hover = index;
    },
    set_first_projection(state, embedding) {
      console.log("commit", embedding)
      state.first_projection = embedding;
      console.log(state.first_projection)
    },
    set_data(state, datasets_index) {
      state.data = datasets_index;
    },
    set_values(state, values) {
      state.values = values;
    },
    set_labels(state, labels) {
      state.labels = labels;
    },
    set_points(state, points) {
      state.points = points;
    },
    set_dimensions(state, dimensions) {
      state.dimensions = dimensions;
    },
    set_pool(state, pool) {
      state.pool = pool;
    },
    set_subspaces(state, subspaces) {
      state.subspaces = subspaces;
    },
    set_projections(state, projections) {
      state.projections = projections;
    },
    set_reorderings(state, reorderings) {
      state.reorderings = reorderings;
    },
    set_comparisons(state, comparisons) {
      state.comparisons = comparisons;
    },
    async set_pointtype(state, pointtype) {
      state.appearance.pointtype = pointtype;
      
      /* if (state.dimensions) {
        state.images = await create_images(state.appearance, state.labels, state.label_color, state.values, state.dimensions)
      } */
    },
    set_scale(state, scale) {
      state.appearance.scale = scale;
    },
    set_sparse(state, v) {
      state.data.sparse = v;
    },  
    set_width(state, width) {
      state.appearance.width = width;
    },
    set_height(state, height) {
      state.appearance.height = height;
    },
    set_orientation(state, orientation) {
      state.appearance.orientation = orientation;
    },
    change_status(state, [projectionId, projectionStatus]) {
      const pool = state.pool;
      pool.queue(worker => worker.set_status_of_projection(projectionId, projectionStatus));
    },
    change_selection(state, array) {
      state.selection = array;
    },
    create_subspace(state, [name, cols]) {
      console.log("store", cols)
      const dimensions = state.dimensions.map((d,i) => {
        return cols.findIndex(c => i === c.index) >= 0;
      })
      const pool = state.pool;
      pool.queue(async worker => {
        const subspace = await worker.create_subspace(name, dimensions)
        state.subspaces.push(subspace)
      });
    },
    set_label_color(state, [type, color]) {
      console.log(type, color)
      switch(type) {
        case "categorical":
          const cat = d3.scaleOrdinal(d3.schemeTableau10)
          state.label_color = (d, i) => cat(d);
          break;
        case "single hue":
          state.label_color = () => color
          break;
        case "rainbow":
          let N = state.points.length;
          const rainbow = d3.scaleSequential(d3.interpolateTurbo)
          state.label_color = (d, i) => rainbow(i / N);
          break;
      }
    },
    set_loading(state, value) {
      state.loading = value;
    }
  },
  actions: {
    async load_data({ state, commit, dispatch }) {
      commit("set_loading", "loading data...");
      commit("set_subspaces", []);
      commit("set_projections", []);
      commit("set_comparisons", []);
      commit("set_first_projection", null);
      const data = state.data;
      state.data_name = data.name;
      let promise = null;
      if (data.path) {
        promise = new Promise((resolve) => {
          d3.csv(data.path).then(data => resolve(data));
        })
      } else if (data.file) {
        promise = new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => {
            const csv_content = reader.result;
            const csv = d3.csvParse(csv_content);
            resolve(csv);
          }
          reader.readAsText(data.file);
        })
      }
      const result = await promise;
      state.data.size = [result.length, result[0].length];
      await dispatch("init_pool", result);
      //commit("set_loading", "caching images of points...");
      state.appearance = data.appearance;
      commit("set_label_color", [state.data.appearance.colorscale, "#daa520"])
    },
    async init_pool({ commit, state }, result) {
      commit("set_loading", "initialize worker pool...");
      if (state.pool) await state.pool.terminate(true);
      let workers = state.workers = [];
      const pool = state.pool = await Pool(async () => {
        const worker = await spawn(new Worker("../../public/js/worker.js"));
        workers.push(worker)
        return worker;
      }, {"concurrency": 2})
      pool.queue(async (worker) => {
        commit("set_loading", "distribute data to the workers...");
        const O = await worker.convert_data(state.data, result)
        state.dimensions = O.dimensions;
        state.points = O.names;
        state.values = O.values;
        commit("change_selection", O.names.map(() => true))
        state.labels = O.labels; 
        //state.label_color = d3.scaleOrdinal(d3.schemeTableau10);
        //commit("set_label_color", state.data.appearance.colorscale)
        const subspaces = await worker.get_subspaces();
        const projections = await worker.get_projections();
        const reorderings = await worker.get_reorderings();
        reorderings.forEach(reorder => {
          reorder.left_projection = projections.find(p => p.id == reorder.left_projectionId)
          reorder.right_projection = reorder.right_projectionId != 'null' ?
            projections.find(p => p.id == reorder.right_projectionId) : null;
          
        })
        commit("set_loading", "start app...");
        commit("set_subspaces", subspaces);
        commit("set_projections", projections);
        commit("set_reorderings", reorderings);
        workers.forEach(w => {
          w.set_data(O);
        })
        commit("set_loading", null);
      });
    },
    add_to_datasets({ state }, file) {
      if (!file) return;
      state.datasets.datasets.push({
        "name": file.name.slice(0, -4),
        "file": file,
        "appearance": {
          "pointtype": "point",
          "scale": 1,
          "width": 1,
          "height": 1,
          "colorscale": "categorical",
        }
      })
    },
    send_to_worker({ state }, [list, subspace]) {
      const pool = state.pool;
      console.log("send_to_worker", list, subspace)
      list.forEach(projection => {
        const thread = pool.queue(async worker => {
          const result = await worker.compute_projection(projection, subspace);
          console.log(result)
          state.projections.push(result);
          const t_i = state.threads.findIndex(d => d.thread == thread);
          state.threads.splice(t_i, 1);
        })
        state.threads.push({
          "thread": thread,
          "projection": projection,
        })
        //console.log(thread)
      })
    },
    async draw_row({ state }, [projection, left_canvas, right_canvas, width]) {
      const pool = state.draw_pool;
      const thread = pool.queue(async worker => {
        const [l, r] = await worker.draw(projection, Transfer(left_canvas), Transfer(right_canvas), width, window.devicePixelRatio)
      })
      await pool.completed();
    },
    async draw_comparison({ state }, [left_projection, right_projection, left_canvas, mid_canvas, right_canvas, width]) {
      const draw_pool = state.draw_pool;
      draw_pool.queue(async worker => {
        await worker.draw_comparison(left_projection, right_projection, Transfer(left_canvas), Transfer(mid_canvas), Transfer(right_canvas), width, window.devicePixelRatio);
      })
    },
    delete_comparison({ state }, [left_projection, right_projection]) {
      let comparisons = state.comparisons;
      const i = comparisons.findIndex(c => {
        return c.left_projection == left_projection && c.right_projection == right_projection;
      })
      comparisons = comparisons.splice(i, 1);
    },
    async reorder_matrix({ state }, [left_projection, right_projection, type]) {
      const left = left_projection.id;
      const right = right_projection ? right_projection.id : "null";
      const pool = state.pool;
      let reordering;
      if (type == "none") {
        reordering = state.points.map((_, i) => i);
      } else {
        const { reordering: result } = await pool.queue(async worker => await worker.reorder_matrix(left, right, type))
        reordering = result;
      }
      const element = {
        "left_projection": left_projection,
        "right_projection": right_projection,
        "type": type,
        "reordering": reordering
      }
      if (state.reorderings.findIndex(r => r.left_projection == left_projection && r.right_projection == right_projection && r.type == type) < 0) {
        if (type !== "none") state.reorderings.push(element)
      }
      console.log(element)
      return element;
    },
    async reorder_matrix_within({ state }, [left_projection, right_projection, type]) {
      const left = left_projection.id;
      const right = right_projection ? right_projection.id : "null";
      const pool = state.pool;
      let reordering;
      if (type == "none") {
        reordering = state.points.map((_, i) => i);
      } else {
        const { reordering: result } = await pool.queue(async worker => await worker.reorder_matrix_within(left, right, type))
        reordering = result;
      }
      const element = {
        "left_projection": left_projection,
        "right_projection": right_projection,
        "type": type,
        "reordering": reordering
      }
      if (state.reorderings.findIndex(r => r.left_projection == left_projection && r.right_projection == right_projection && r.type == type) < 0) {
        if (type !== "none") state.reorderings.push(element)
      }
      console.log(element)
      return element;
    },    
    async delete_db({ state }) {
      const pool = state.pool;
      return await pool.queue(async worker => {
        await worker.delete_db();
      })
    },
    async get_db({ state }) {
      const pool = state.pool;
      const db_blob = await pool.queue(async worker => await worker.get_db());
      download(db_blob, "indexed.db", "plain/text")
    },
    async load_db({ state }, db_blob) {
      console.log(db_blob)
      const pool = state.pool;
      await pool.queue(async worker => await worker.load_db(db_blob));
    }
  }
})



/* function procrust(source, other) {
  //translate + scale
  [source, other].forEach(data => {
      const x_mean = d3.mean(data, d => d[0])
      const y_mean = d3.mean(data, d => d[1])
      let scale_factor = 0;
      let n = data.length;
      for (let i = 0; i < n; ++i) {
          scale_factor += (data[i][0] * data[i][0] + data[i][1] * data[i][1])
      }
      scale_factor = Math.sqrt(scale_factor / n);
      for (let i = 0; i < n; ++i) {
          data[i][0] = (data[i][0] - x_mean) / scale_factor
          data[i][1] = (data[i][1] - y_mean) / scale_factor
      }
  })

  // rotate
  let n = other.length;
  let t = 0, b = 0;
  for (let i = 0; i < n; ++i) {
      t += (source[i][1] * other[i][1] - source[i][0] * other[i][0])
      b += (source[i][1] * other[i][1] + source[i][0] * other[i][0])
  }
  let phi = Math.atan(t/b);
  let cos_a = Math.cos(phi)
  let sin_a = Math.sin(phi)
  for (let i = 0; i < n; ++i) {
      const [x, y] = other[i];
      other[i] = [
          x * cos_a + y * sin_a,
          x * -sin_a + y * cos_a
      ]
  }

  return other;
} */