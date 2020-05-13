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
                visibility: drawn ? 'unset' : 'hidden',
            }"
            ref="canvas"
            >

        </canvas>
        <svg
            :viewBox="`0 0 ${width} ${height}`"
            :width="width"
            :height="height"
            ref="svg"
            :style="{
                visibility: drawn ? 'unset' : 'hidden',
                }">
            <g v-if="!horizontal" >
                <text
                    :x="margin * 14" 
                    :y="svg_y.range()[0]" 
                    :transform="`rotate(90 ${margin * 14},${svg_y.range()[0]})`"
                    text-anchor="start">
                    {{ right_projection ? "closer on the left projection" : "closer in the projection" }}
                </text>
                <text 
                    :x="margin * 14" 
                    :y="svg_y.range()[1]" 
                    :transform="`rotate(90 ${margin * 14},${svg_y.range()[1]})`"
                    text-anchor="end">
                    {{ right_projection ? "closer on the right projection" : "farther in the projection" }}
                </text>
            </g>
            <g v-else>
                <text
                    :y="margin * 14" 
                    :x="svg_y.range()[0]" 
                    text-anchor="start">
                    {{ right_projection ? "closer on the left projection" : "closer in the projection" }}
                </text>
                <text 
                    :y="margin * 14" 
                    :x="svg_y.range()[1]" 
                    text-anchor="end">
                    {{ right_projection ? "closer on the right projection" : "farther in the projection" }}
                </text>
            </g>
            <g v-if="!horizontal" :transform="`translate(${margin * 6}, 0)`" id="axis"></g>
            <g v-else :transform="`translate(0, ${margin * 6})`" id="axis"></g>
        </svg>
    </div>
</template>

<script>
import * as d3 from "d3";
import { mapState } from "vuex";

export default {
    name: "detail-canvas",
    props: ["left_projection", "right_projection", "width", "height", "horizontal"],
    data: function() {
        return {
            dpr: (window.devicePixelRatio || 1),
            visible: true,
            margin: 10 / (window.devicePixelRatio || 1),
            drawn: false,
        }
    },
    watch: {
        width: async function(n, o) {
            this.width = n
            await this.draw()
        }
    },
    computed: {
        ...mapState({
            worker: state => state.workers[0],
        }),
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
        x: async function() {
            const dpr = this.dpr;
            const matrix = await this.matrix;
            const N = d3.max(await this.bins, d => d.length);
            const margin = this.margin * dpr;
            const width = this.width * dpr;
            return d3.scaleLinear()
                .domain([0, N])
                .range(["transparent", "#8888"])
                //.range([1 * margin, (width - margin) / 2])
        },
        y: function() {
            const dpr = this.dpr;
            const height = this.height * dpr;
            const width = this.width * dpr;
            const margin = this.margin * dpr;
            return d3.scaleLinear()
                .domain([-1, 1])
                .range([margin, (this.horizontal ? width : height) - margin]);
        },
        svg_y: function() {
            const height = this.height;
            const width = this.width;
            const margin = this.margin;
            return d3.scaleLinear()
                .domain([-1, 1])
                .range([margin, (this.horizontal ? width : height) - margin]);
        },
        bins: async function() {
            const matrix = await this.matrix;
            //const x = await this.x;
            const y = this.y;
            return d3.histogram()
                .domain(y.domain())
                .thresholds(y.ticks(40))(matrix.flat());
        },
    },
    mounted: async function() {
        await this.draw();
        this.drawn = true;
    },
    methods: {
        visibility_changed: async function(visible) {
            this.visible = visible;
        },
        draw: async function() {
            const horizontal = this.horizontal;
            const y = this.y;
            const canvas = this.$refs.canvas;
            const context = canvas.getContext("2d");
            const dpr = this.dpr;
            const width = this.width * dpr;
            const height = this.height * dpr;
            const margin = this.margin * dpr;
            if (horizontal) {
                const gradient = context.createLinearGradient(0,0,width - 2 * margin, 0)
                gradient.addColorStop(1, "tomato");
                gradient.addColorStop(0.5, "transparent");
                gradient.addColorStop(0, "steelblue");
                context.clearRect(0, 0, width, height);
                context.strokeStyle = gradient;
                context.lineCap = "round"
                context.lineWidth = margin;
                //context.fillRect(margin, margin, margin, height - 2 * margin)
                context.beginPath();
                context.moveTo(y(-1) + 2 * margin, margin);
                context.lineTo(y(1) - 2 * margin, margin);
                context.stroke();
            } else {
                const gradient = context.createLinearGradient(0,0,0,height - 2 * margin)
                gradient.addColorStop(1, "tomato");
                gradient.addColorStop(0.5, "transparent");
                gradient.addColorStop(0, "steelblue");
                context.clearRect(0, 0, width, height);
                context.strokeStyle = gradient;
                context.lineCap = "round"
                context.lineWidth = margin;
                //context.fillRect(margin, margin, margin, height - 2 * margin)
                context.beginPath();
                context.moveTo(margin, y(-1));
                context.lineTo(margin, y(1));
                context.stroke();
            }

            const bins = await this.bins;
            const x = await this.x;
            /* for (let i = 0; i < bins.length; ++i) {
                const bin = bins[i]
                context.fillStyle = "lightgrey";
                const y_pos = y(bin.x0)
                const h = y(bin.x1) - y(bin.x0);
                const w = x(bin.length)
                context.fillRect(x(0) - margin, y_pos, w - 2 * margin, h);
            }   */  
            const svg = d3.select(this.$refs.svg);
            const svg_y = this.svg_y;
            svg.selectAll("rect")
                .data(bins)
                .enter()
                    .append("rect")
                .join(svg.selectAll("rect"))
                    .attr(horizontal ? "x" : "y", d => svg_y(d.x0))
                    .attr(horizontal ? "y" : "x", margin)
                    .attr(horizontal ? "width" : "height", d => (svg_y(d.x1) - svg_y(d.x0)))
                    .attr(horizontal ? "height" : "width", 2 * margin)
                    .attr("fill", d => x(d.length))//"lightgrey")

            svg.select("#axis")
                .call(horizontal ? d3.axisBottom(this.svg_y) : d3.axisRight(this.svg_y))

            //svg.select("#axis").select(".domain").attr("stroke", "transparent")

        },
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

    text {
        font-family: sans-serif;
        font-size: 10px;
        fill: currentColor;
    }

</style>