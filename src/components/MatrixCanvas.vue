<template>
    <div v-observe-visibility="{
            'callback': visibility_changed,
            'throttle': 50,
        }">

        <v-skeleton-loader 
            :style="{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                }"
            tile
            v-if="!drawn" 
            :width="width" 
            :height="height" 
            type="image, image">
            
        </v-skeleton-loader>

        <canvas 
            :width="width * dpr"
            :height="height * dpr"
            :style="{
                width: width + 'px',
                height: height + 'px',
            }"
            ref="canvas">
        </canvas>

        <svg
            :viewBox="`0 0 ${width} ${height}`"
            :width=width
            :height=height

            @mousemove="mousemove"
            @mouseleave="mouseleave"
            ref="svg">
        </svg>
    </div>
</template>

<script>
import * as d3 from "d3";
import { mapState } from "vuex";

export default {
    name: "matrix-canvas",
    props: [
        "left_projection", 
        "right_projection", 
        "width", 
        "height", 
        "reordering", 
    ],
    data: function() {
        return {
            visible: true,
            margin: 10 / (window.devicePixelRatio || 1),
            dpr: (window.devicePixelRatio || 1),
            o: i => i,
            brush: d3.brush(),
            drawn: false,
            local_selection: null,
        }
    },
    watch: {
        selection: function() {
            d3.select(this.$refs.svg).call(this.brush.move, null)
            this.draw();
        },
        local_selection: function() {
            this.draw();
        }
    },
    computed: {
        ...mapState({
            worker: state => state.workers[(state.workers.length - 3) % state.workers.length],
            labels: state => state.labels,
            label_color: state => state.label_color,
        }),
        tooltip: {
            get: function() {
                return this.$store.state.tooltip;
            },
            set: function(msg) {
                this.$store.commit("set_tooltip", msg);
            }
        },
        hover: {
            get: function() {
                return this.$store.state.hover;
            },
            set: function(index) {
                this.$store.commit("set_hover_index", index);
            }
        },
        selection: {
            get: function() {
                return this.$store.state.selection;
            },
            set: function(array) {
                this.$store.commit("change_selection", array)
            }
        },
        matrix: async function() {
            const worker = this.worker;
            const left_projection = this.left_projection;
            const right_projection = this.right_projection;
            const left = left_projection.id;
            const right = right_projection ? right_projection.id : "null";
            return await worker.get_discrepancies(left, right);
        },
        extent: async function() {
            return d3.extent((await this.matrix).flat());
        },
        scales: async function() {
            const [min, max] = await this.extent;
            const neg_scale = min < 0 ? 
                d3.scaleLinear()
                    .domain([-1, 0])
                    .range(["steelblue", "transparent"]) 
                :
                () => "transparent";
            const pos_scale = max > 0 ? 
                d3.scaleLinear()
                    .domain([0, 1])
                    .range(["transparent", "tomato"]) 
                :
                () => "transparent";
            return {
                neg_scale: neg_scale,
                pos_scale: pos_scale,
            }
        },
        x: async function() {
            const matrix = await this.matrix;
            const N = matrix.length;
            const margin = this.margin;
            const width = this.width;
            return d3.scaleLinear()
                .domain([0, N]).range([3 * margin, width - margin])
        }

    },
    mounted: async function() {
        const brush = this.brush;
        const width = this.width;
        const height = this.height;
        const margin = this.margin;

        brush.extent([[3 * margin, 3 * margin], [width - margin, height - margin]])
            .on("brush", () => this.brushed(d3.event))
            .on("end", async () => await this.brushend(d3.event))

        d3.select(this.$refs.svg).call(this.brush)

        d3.select(this.$refs.svg).select(".selection")
            .attr("fill-opacity", .1)
            .attr("stroke", "grey")
            .attr("stroke-dasharray", 3)
            .style("animation", "dash 5s linear infinite");
        
        await this.draw();
        this.drawn = true;
    }, 
    methods: {
        draw: async function() {
            if (!this.visible) return;
            //this.drawn = true;
            const matrix = await this.matrix;
            const { neg_scale, pos_scale } = await this.scales;
            const N = matrix.length;
            const canvas = this.$refs.canvas;
            const context = canvas.getContext("2d");
            const dpr = this.dpr;
            const width = this.width;
            const labels = this.labels;
            const color = this.label_color;
            const margin = this.margin;

            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0, 0, width, width);

            const x = await this.x;
            const w = x(1) - x(0);
            const o = this.o;
            const s = this.local_selection || this.selection;
            //console.log("matrix draw", this.local_selection)
            for (let i = 0; i < N; ++i) {
                const xi = x(o(i))
                const label_color = d3.hsl(color(labels[i], i))
                if (!s[i]) label_color.s = 0;
                context.fillStyle = label_color.toString();
                context.fillRect(xi, margin, w, margin);
                context.fillRect(margin, xi, margin, w);
                for (let j = i + 1; j < N; ++j) {
                    const xj = x(o(j))
                    const val = matrix[i][j];
                    let c = d3.hsl(val < 0 ? neg_scale(val) : pos_scale(val));
                    if (!s[i] && !s[j]) {
                        c.s = 0;
                        c.l = .4;
                    }
                    context.fillStyle = c.toString();
                    context.fillRect(xi, xj, w, w);
                    context.fillRect(xj, xi, w, w);
                }
            }

            /* const h = this.hover;
            if (h) {
                context.fillStyle = "black";
                context.fillRect(x(o(h)), 0, 1, margin)
                context.fillRect(0, x(o(h)), margin, 1)
            } */
            this.drawn = true;
        },
        visibility_changed: async function(visible) {
            this.visible = visible;
            /* if (visible) this.draw(); */
        },
        mousemove: async function(event) {
            const x = await this.x;
            const {offsetX, offsetY} = event;
            const pos = Math.round(x.invert(offsetX))
            const N = this.$store.state.points.length
            const index = Math.max(0, Math.min(N, pos));
            const o = this.o;
            this.tooltip = {
                top: event.clientY,
                left: event.clientX,
            }
            //console.log(index)
            this.hover = o(index);
        },
        mouseleave: function(event) {
            this.tooltip = null;
            this.hover = null;
        },
        brushed: function(event) {
            if (event.selection == null || (event.sourceEvent && event.sourceEvent.type == "brush")) return;
            const sel = event.selection;
            //if (sel[0][0] > sel[1][0]) sel[1][0] = sel[0][0];
            sel[0][1] = sel[0][0];
            sel[1][0] = sel[1][1];
            d3.select(this.$refs.svg).call(this.brush.move, sel);
        },
        brushend: async function(event) {
            //console.log(event.sourceEvent)
            if (event.selection == null || (event.sourceEvent && event.sourceEvent.type == "end")) return;
            
            const x = await this.x;
            const sel = event.selection;
            //if (sel[0][0] > sel[1][0]) sel[1][0] = sel[0][0];

            sel[0][1] = sel[0][0];
            sel[1][1] = sel[1][0];

            //d3.select(this.$refs.svg).call(this.brush.move, sel);
            const [[x0, y0], [x1, y1]] = sel.map((d) => [x.invert(d[0]), x.invert(d[1])]);
            const o = this.o;
            /* this.local_selection = this.selection.map((_, noi) => {
                const i = o(noi);
                return (i >= x0 && i <= x1 && i >= y0 && i <= y1)
            }); */
            this.$emit("selection_end", this.selection.map((_, noi) => {
                const i = o(noi);
                return (i >= x0 && i <= x1 && i >= y0 && i <= y1)
            }));
        },
        set_reordering: async function(reordering) {
            this.reordering = reordering;
            this.o = i => reordering.findIndex(d => d ==i);
            await this.draw();
        },
        move_brush_top_left: async function(sel) {
            if (!sel) return;
            const N = sel.length;
            const x = await this.x;

            d3.select(this.$refs.svg)
                .call(this.brush.move, [[x(0), x(0)], [x(N), x(N)]]);
        }
    }
}
</script>

<style>
    @keyframes dash {
        to {
            stroke-dashoffset: 60;
        }
    }
</style>

<style scoped>
    canvas {
        border: 1px solid #aaa2;
        border: 1px solid transparent;
    }

    div {
        position: relative;
    }

    svg {
        border: 1px solid transparent;
        position: absolute;
        top: 0px;
        left: 0px;
    }
</style>