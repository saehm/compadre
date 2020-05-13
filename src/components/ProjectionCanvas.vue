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
                    visibility: drawn,
                }"
                ref="canvas">
            </canvas>
        <svg :viewBox="`0 0 ${width} ${height}`"
            :width=width
            :height=height
            ref="svg"
            @mousemove="mousemove"
            @mouseleave="hover = null; tooltip = null; draw();">

            <g class="lasso">
                <path ref="lasso" class="lasso"></path>
                <path ref="close" class="close"></path>
            </g>
            <rect 
                :width=width 
                :height=height 
                fill="transparent"
                @pointerdown.prevent="(e) => dragstart(e)"
                @pointermove.prevent="(e) => drag(e)"
                @pointerout.prevent
                @pointerup.prevent="dragend"
            >
            </rect>
            <!-- -->
        </svg>
        
    </div>
</template>

<script>
import * as d3 from "d3";
import { mapState } from "vuex";

export default {
    name: "projection-canvas",
    props: ["projection", "width", "height"],
    data: function() {
        return {
            visible: true,
            margin: 10,
            dpr: (window.devicePixelRatio || 1),
            lasso: [],
            lasso_draw: false,
            lasso_selection: null,
            drawn: false,
            local_selection: null,
        }
    },
    watch: {
        selection: function() {
            this.draw();
        },
        hover: function() {
            this.draw();
        },
        lasso_selection: function(n, o) {
            //this.$emit("selection_end", this.lasso_selection);
            this.draw();
        },
        local_selection: function() {
            this.draw();
        }
    },
    computed: {
        ...mapState({
            worker: state => state.workers[(state.workers.length - 1) % state.workers.length],
            labels: state => state.labels,
            label_color: state => state.label_color,
            images: state => state.images,
        }),
        selection: {
            get: function() {
                return this.$store.state.selection;
            },
            set: function(array) {
                this.$store.commit("change_selection", array)
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
        tooltip: {
            get: function() {
                return this.$store.state.tooltip;
            },
            set: function(msg) {
                this.$store.commit("set_tooltip", msg);
            }
        },
        first_projection: {
            get: function() {
                return this.$store.state.first_projection;
            },
            set: function(embedding) {
                this.$store.commit("set_first_projection", embedding);
            }
        }, 
        embedding: async function() {
            const worker = this.worker;
            let embedding = await worker.get_embedding(this.projection.id);
            let first_projection = this.first_projection;
            if (first_projection) {
                embedding.projection = this.procrust([first_projection.projection, embedding.projection]);
            } else {
                this.first_projection = embedding;
            }
            return embedding;
        },
        extent: async function() {
            const {projection: embedding} = await this.embedding;
            const x_extent = d3.extent(embedding, d => d[0]);
            const y_extent = d3.extent(embedding, d => d[1]);
            const x_span = x_extent[1] - x_extent[0];
            const y_span = y_extent[1] - y_extent[0];
            const offset = Math.abs(x_span - y_span) / 2;

            if (x_span > y_span) {
                y_extent[0] -= offset;
                y_extent[1] += offset;
            } else {
                x_extent[0] -= offset;
                y_extent[1] += offset;
            }

            return {
                "x_extent": x_extent,
                "y_extent": y_extent,
            };

        },
        scales: async function() {
            const extent = await this.extent;
            const width = this.width;
            const height = this.height;
            const margin = this.margin;
            const x = d3.scaleLinear()
                .domain(extent.x_extent)
                .range([margin, width - margin]);
            const y = d3.scaleLinear()
                .domain(extent.y_extent)
                .range([margin, height - margin]);
            return {
                "x": x,
                "y": y,
            }
        },
        quadtree: async function() {
            const {x, y} = await this.scales;
            const {projection: embedding} = await this.embedding;
            return d3.quadtree()
                .x(d => x(d[0]))
                .y(d => y(d[1]))
                .addAll(embedding);
        },

    },
    methods: {
        draw: async function() {
            //if (!this.visible) return;
            const {projection: embedding} = await this.embedding;
            const {x: x, y: y} = await this.scales;
            const canvas = this.$refs.canvas;
            const dpr = this.dpr;
            const labels = this.labels;
            const color = this.label_color;
            this.drawn = true;
            const context = canvas.getContext("2d");
            const width = this.width;
            const hover = this.hover;
            const selection = this.local_selection || this.selection;
            //console.log("projection draw", this.local_selection)
            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0,0,width, width);
            const images = this.images;
            if (images) {
                const {width: w, height: h, scale} = this.$store.state.appearance;
                embedding.forEach(([px, py], i) => {
                    const image = images[i];
                    if (image) {
                        context.filter = `grayscale(${selection[i] ? 0 : 1})`
                        if (hover != null && hover == i) {
                            context.drawImage(image, x(px) - w * (scale * 1.5) / 2, y(py) - h * (scale * 1.5) / 2, w * (scale * 1.5), h * (scale * 1.5))
                        } else {
                            context.drawImage(image, x(px) - w * scale / 2, y(py) - h * scale / 2, w * scale, h * scale)
                        }
                    }
                    
                })
            } else {
                const scale = this.$store.state.appearance.scale;
                embedding.forEach((point, i) => {
                    const px = x(point[0]);
                    const py = y(point[1]);
                    context.beginPath();
                    context.arc(px, py, ((hover && hover == i) ? 4 : 2) * scale, 0, Math.PI * 2);
                    context.strokeStyle = selection[i] ? color(labels[i], i) : "grey"
                    context.lineWidth = (hover && hover == i) ? 2 : 1;
                    context.stroke();
                    if (hover && hover == i) {
                        context.fillStyle = selection[i] ? color(labels[i], i) : "grey"
                        context.fill();
                    }
                })
            }
            this.drawn = true;
        },
        visibility_changed: async function(visible) {
            this.visible = visible;
            if (visible) await this.draw();
        },
        mousemove: async function(event) {
            const {projection: embedding} = await this.embedding;
            const quadtree = await this.quadtree;
            const pos_x = event.offsetX;
            const pos_y = event.offsetY;
            const point = quadtree.find(pos_x, pos_y, 20);
            if (point != undefined) {
                this.hover = embedding.findIndex(d => d == point)
                this.tooltip = {
                    top: event.clientY,
                    left: event.clientX,
                }
            } else {
                this.hover = null;
                this.tooltip = null;
            }
            await this.draw();
        },
        dragstart: async function(event) {
            this.lasso_draw = true;
            const {offsetX: x, offsetY: y} = event;
            const lasso = this.lasso = [];
            lasso.push([x, y])
            const lasso_path = `M${lasso.map(d => d.join(',')).join('L')}`;
            this.$refs.lasso.setAttribute("d", lasso_path);

            const {projection: embedding} = await this.embedding;
            this.lasso_selection = embedding.map(() => false);
        },
        drag: async function(event) {
            if (!this.lasso_draw) return;
            const {offsetX: x, offsetY: y} = event
            const lasso = this.lasso;
            lasso.push([x, y])
            const lasso_path = `M${lasso.map(d => d.join(',')).join('L')}`;
            this.$refs.lasso.setAttribute("d", lasso_path);
            const {x: xs, y: ys} = await this.scales;
            const {projection: embedding} = await this.embedding;
            this.lasso_selection = embedding.map(d => d3.polygonContains(lasso, [xs(d[0]), ys(d[1])]));
            //this.$emit("brushing", this.lasso_selection);
        },
        dragend: async function(event) {
            if (!this.lasso_draw) return;
            this.lasso_draw = false;
            const {offsetX: x, offsetY: y} = event
            const lasso = this.lasso;
            lasso.push([x, y])
            //this.$refs.lasso.setAttribute("d", "");
            const lasso_path = `M${lasso.map(d => d.join(',')).join('L')}`;
            this.$refs.lasso.setAttribute("d", lasso_path);
            
            const {x: xs, y: ys} = await this.scales;
            const {projection: embedding} = await this.embedding;
            /* this.lasso_selection = null
            this.selection = embedding.map(d => d3.polygonContains(lasso, [xs(d[0]), ys(d[1])])); */
            this.lasso_selection = embedding.map(d => d3.polygonContains(lasso, [xs(d[0]), ys(d[1])]));
            this.$emit("selection_end", this.lasso_selection);
        },
        procrust: function(embeddings) {
            let [source, other] = embeddings;
            //translate + scale
            /* embeddings.forEach(data => {
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
            }) */

            // rotate
            /* let n = other.length;
            let t = 0, b = 0;
            for (let i = 0; i < n; ++i) {
                t += (source[i][1] * other[i][1] - source[i][0] * other[i][0])
                b += (source[i][1] * other[i][1] + source[i][0] * other[i][0])
            }
            let phi = Math.atan(t/b) / Math.PI * 180;
            let cos_a = Math.cos(phi)
            let sin_a = Math.sin(phi)
            for (let i = 0; i < n; ++i) {
                const [x, y] = other[i];
                other[i] = [
                    x * cos_a + y * sin_a,
                    x * -sin_a + y * cos_a
                ]
            } */

            return other;
        }
    }, 
    mounted: async function() {
        await this.draw();
        this.drawn = false;
    }
}
</script>

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

    .lasso {
        fill-opacity: .1;
        stroke: grey;
        stroke-dasharray: 3;
        animation: dash 5s linear infinite;
    }

    @keyframes dash {
        to {
            stroke-dashoffset: 60;
        }
    }

    .close {
        stroke: grey;
        stroke-dasharray: 2 2;
    }
</style>