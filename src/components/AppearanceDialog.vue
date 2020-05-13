<template>
    <v-dialog 
        scrollable
        v-model="dialog" 
        max-width="600px">
        <template v-slot:activator="{ on }">
            <!-- <v-list-item v-on="on">
                <v-list-item-action>
                    <v-icon>mdi-pencil</v-icon>
                </v-list-item-action>
                <v-list-item-content>
                    <v-list-item-title>
                    Appearance
                    </v-list-item-title>
                </v-list-item-content>
            </v-list-item> -->

            <v-btn 
                :disabled="loading != null" 
                icon
                v-on="on">
                <v-tooltip bottom>
                    <template v-slot:activator="{ on }">
                        <v-icon v-on="on">mdi-shape</v-icon>
                    </template>
                    <span>Appearance</span>
                </v-tooltip>
            </v-btn>
            <!-- <v-btn text v-on="on">Appearance</v-btn>
            <v-btn icon v-on="on"><v-icon>mdi-shape</v-icon></v-btn> -->
        </template>
        <v-card :loading="loading">
            <v-toolbar class="mb-6">
                <v-toolbar-title>
                    Appearance
                </v-toolbar-title>
                <template slot="extension">
                    <v-tabs
                        align-with-title
                        show-arrows
                        v-model="tab">
                        <v-tab>Point</v-tab>
                        <v-tab>Parallel Coordinates</v-tab>
                        <v-tab>Image</v-tab>
                    </v-tabs>
                </template>

            </v-toolbar>
            <v-card-text>
                <v-tabs-items 
                    v-model="tab"
                    class="mb-4">
                    
                    <v-form ref="point_form">
                        <v-tab-item>
                            <v-slider 
                                v-model="scale" 
                                min=.1
                                max=2
                                step=.1
                                ticks
                                label="Scale" 
                                prepend-icon="mdi-image-size-select-small">
                                <template v-slot:append>
                                    <v-text-field 
                                        v-model="scale" 
                                        class="mt-0 pt-0" 
                                        hide-details 
                                        single-line 
                                        min=.1
                                        max=2
                                        step=.1
                                        type="number" 
                                        style="width: 60px">
                                    </v-text-field>
                                </template>
                            </v-slider>
                        </v-tab-item>
                    </v-form>
                    <v-form ref="pcp_form">
                        <v-tab-item>
                            <v-row>
                                <v-col sm=12>
                                    <v-slider 
                                        v-model="scale" 
                                        min=.1
                                        max=2
                                        step=.1
                                        ticks
                                        label="Scale" 
                                        prepend-icon="mdi-image-size-select-small">
                                        <template v-slot:append>
                                            <v-text-field 
                                                v-model="scale" 
                                                class="mt-0 pt-0" 
                                                hide-details 
                                                single-line 
                                                min=.1
                                                max=2
                                                step=.1
                                                type="number" 
                                                style="width: 60px">
                                            </v-text-field>
                                        </template>
                                    </v-slider>
                                </v-col>
                                <v-col sm=6>
                                    <v-text-field 
                                        prepend-icon="mdi-arrow-expand-horizontal"
                                        label="Width"
                                        v-model="width" 
                                        suffix="px" 
                                        class="mt-0 pt-0" 
                                        step=1
                                        type="number">
                                    </v-text-field>
                                </v-col>
                                <v-col sm=6>
                                    <v-text-field 
                                        prepend-icon="mdi-arrow-expand-vertical"
                                        label="Height"
                                        v-model="height" 
                                        suffix="px" 
                                        class="mt-0 pt-0" 
                                        step=1
                                        type="number">
                                    </v-text-field>
                                </v-col>
                                <v-col sm=12>
                                    <v-radio-group ref="orientation_switch"
                                        v-model="orientation"
                                        label="Orientation"
                                        mandatory
                                        prepend-icon="mdi-rotate-right"
                                        row color="primary">
                                        <v-radio color="primary" label="horizontal" value="horizontal"></v-radio>
                                        <v-radio color="primary" label="vertical" value="vertical"></v-radio>
                                        <v-radio color="primary" label="radial" value="radial"></v-radio>
                                    </v-radio-group>
                                </v-col>
                            </v-row>
                        </v-tab-item>
                    </v-form>
                    <v-form ref="image_form">
                        <v-tab-item>
                            <v-row>
                                <v-col sm=12>
                                    <v-slider 
                                        v-model="scale" 
                                        min=.1
                                        max=2
                                        step=.1
                                        ticks
                                        label="Scale" 
                                        prepend-icon="mdi-image-size-select-small">
                                        <template v-slot:append>
                                            <v-text-field 
                                                v-model="scale" 
                                                class="mt-0 pt-0" 
                                                hide-details 
                                                single-line 
                                                min=.1
                                                max=2
                                                step=.1
                                                type="number" 
                                                style="width: 60px">
                                            </v-text-field>
                                        </template>
                                    </v-slider>
                                </v-col>
                                <v-col sm=6>
                                    <v-text-field 
                                        prepend-icon="mdi-arrow-expand-horizontal"
                                        label="Width"
                                        v-model="width" 
                                        suffix="px" 
                                        class="mt-0 pt-0" 
                                        hide-details 
                                        single-line
                                        step=1
                                        type="number">
                                    </v-text-field>
                                </v-col>
                                <v-col sm=6>
                                    <v-text-field 
                                        prepend-icon="mdi-arrow-expand-vertical"
                                        label="Height"
                                        v-model="height" 
                                        suffix="px" 
                                        class="mt-0 pt-0" 
                                        hide-details 
                                        single-line 
                                        step=1
                                        type="number">
                                    </v-text-field>
                                </v-col>
                            </v-row>
                        </v-tab-item>
                    </v-form>
                    <v-form ref="extra_form">
                        <v-row>
                            <v-col sm=12>
                                <v-select 
                                    label="Colorscale"
                                    v-model='colorscale'
                                    :items="['categorical', 'single hue', 'rainbow']">

                                </v-select>
                            </v-col>
                            <v-col sm=12>
                                <v-color-picker 
                                    v-if="colorscale == 'single hue'" 
                                    v-model="single_hue"
                                    width="573px"
                                    hide-inputs
                                    flat
                                    @input="set_label_color('single hue')"/>
                            </v-col>
                        </v-row>
                        <v-row v-if="tab != 0">
                            <v-subheader>Example</v-subheader>
                            <v-col ref="example_col" sm=12>
                                <canvas ref="example_canvas"
                                    :style="{ width: '100%', height: (height * scale || 0) + 'px' }"
                                    />
                            </v-col>
                        </v-row>
                    </v-form>
                </v-tabs-items>
                
            </v-card-text>
            <v-divider />
            <v-card-actions>
                <v-spacer />
                <v-btn text @click="dialog = false">Cancel</v-btn>
                <v-btn 
                    depressed 
                    @click="confirm_dialog" 
                    color="primary"
                    :loading="!!loading || drawing">
                    OK
                </v-btn>
            </v-card-actions>

        </v-card>
    </v-dialog>
</template>

<script>
import { mapState, mapActions } from "vuex";
import * as d3 from "d3";

export default {
    name: "appearance-dialog",
    data: function() {
        return {
            drawing: false,
            dialog: false,
            single_hue: "#daa520",
            scale: 1,
            colorscale: null,
            orientation: null,
            width: 0,
            height: 0,
            tab: 0,
            sample: null,
        }
    },
    watch: {
        appearance: async function(n, o) {
            console.log("appearance_dialog", n)
            this.scale = n.scale;
            this.width = n.width;
            this.height = n.height;
            this.colorscale = n.colorscale;
            this.tab = {"point": 0, "parallel coordinates": 1, "image": 2}[n.pointtype];
            this.orientation = n.orientation;
            this.single_hue = "#daa520";
            this.sample = this.get_sample();
        },
        values: function() {
            this.sample = this.get_sample();
        },
        scale: function() {
            this.sample = this.get_sample();
            this.draw_images();
        },
        colorscale: function() {
            this.$store.commit("set_label_color", [this.colorscale, this.single_hue]);
        },
        actual_data: async function() {
            await this.confirm_dialog();
        }
    },
    computed: {
        ...mapState({
            appearance: state => state.appearance,
            values: state => state.values,
            labels: state => state.labels,
            dimensions: state => state.dimensions,
            label_color: state => state.label_color,
            loading: state => state.loading,
            actual_data: state => state.data,
        }),
        images: {
            get: function() {
                return this.$store.state.images;
            },
            set: function(images) {
                this.$store.commit("set_images", images)  
                //this.$store.state.images = images
            }

        },
        pointtype: function() {
            return ["point", "parallel coordinates", "image"][this.tab]
        },
        context: function() {
            return this.$refs.example_canvas.getContext("2d");
        }
    },
    mounted() {
        const n = this.appearance;
        this.scale = n.scale;
        this.width = n.width;
        this.height = n.height;
        this.colorscale = n.colorscale;
        this.tab = {"point": 0, "parallel coordinates": 1, "image": 2}[n.pointtype];
        this.orientation = n.orientation;
        this.single_hue = "#daa520";
        this.sample = this.get_sample();
    },
    methods: {
        async confirm_dialog() {
            this.drawing = true;
            //this.loading = "caching images of points...";
            this.$store.commit("set_width", this.width);
            this.$store.commit("set_height", this.height);
            this.$store.commit("set_scale", this.scale);
            this.$store.commit("set_orientation", this.orientation);
            this.$store.commit("set_pointtype", this.pointtype);
            this.$store.commit("set_label_color", [this.pointtype, this.single_hue]);
            this.$store.commit("set_images", await this.create_images(this.values, this.labels));
            this.drawing = false;
            this.dialog = false;
            //this.loading = null;
        },
        get_sample() {
            const labels = this.labels;
            const values = this.values;
            if (!values) return [];
            const width = this.width * this.scale;
            const canvas = this.$refs.example_canvas;
            const canvas_width = canvas ? canvas.clientWidth : 500;
            const n = Math.floor(canvas_width / width);
            const indices = d3.range(values.length);
            const sample_labels = [];
            const sample_values = [];
            const sample_indices = [];
            for (let i = 0; i < n; ++i) {
                const m = Math.floor(Math.random() * indices.length);
                const index = indices.splice(m, 1);
                sample_labels.push(labels[index]);
                sample_values.push(values[index]);
                sample_indices.push(index)
            };
            this.sample = {
                labels: sample_labels,
                values: sample_values,
                indices: sample_indices,
            }
            return this.sample;
        },
        draw_images: function() {
            if (!this.$refs.example_canvas) return;            
            this.$refs.example_canvas.width = this.$refs.example_canvas.clientWidth;
            this.$refs.example_canvas.height = this.$refs.example_canvas.clientHeight;
            
            
            const sample = this.sample || this.get_sample()
            if (!sample) return;
            const l = sample.labels;
            const v = sample.values;
            const is = sample.indices;

            const context = this.context; //this.$refs.example_canvas.getContext("2d");
            console.log(this.pointtype)
            if (this.pointtype == "image") {
                const s = d3.scaleLinear()
                    .domain(d3.extent(this.values.flat()))
                    .range([0, 1])
                //const example_images = [];
                for (let i = 0, n = l.length; i < n; ++i) {
                    const img = this.draw_mnist_image(l[i], v[i], is[i], s);
                    const w = img.width;
                    const h = img.height;
                    const scale = this.scale;
                    context.drawImage(img, i * w * scale, 0, w * scale, h * scale)
                }
            } else if (this.pointtype == "parallel coordinates" && this.orientation != "radial") {
                const w = this.width;
                const h = this.height;
                const N = this.dimensions.length;
                const y = d3.scaleLinear()
                    .domain([0, N])
                    .range([0, h]);
                const xs = this.dimensions.map((_, i) => {
                    return d3.scaleLinear()
                        .domain(d3.extent(v, d => d[i]))
                        .range([1, w - 1]);
                });
                for (let i = 0, n = l.length; i < n; ++i) {
                    const img = this.draw_pcp_image(l[i], v[i], is[i], y, xs)
                    const w = img.width;
                    const h = img.height;
                    const scale = this.scale;
                    context.drawImage(img, i * w * scale, 0, w * scale, h * scale)
                }

            } else if (this.pointtype == "parallel coordinates" && this.orientation == "radial") {
                const dpr = window.devicePixelRatio || 1;
                const w = this.width;
                const h = this.height;
                const N = this.dimensions.length;
                const angle = d3.scaleLinear()
                    .domain([0, N ])
                    .range([0, 2 * Math.PI]);
                const radii = this.dimensions.map((_, i) => {
                    return d3.scaleLinear()
                        .domain(d3.extent(v, d => d[i]))
                        .range([0, w / 2]);
                });
                for (let i = 0, n = l.length; i < n; ++i) {
                    const img = this.draw_radial_image(l[i], v[i], is[i], angle, radii)
                    const w = img.width;
                    const h = img.height;
                    const scale = this.scale;
                    context.drawImage(img, i * w * scale, 0, w * scale, h * scale)
                }
            }
        },
        draw_mnist_image: function(label, row, index, s) {
            const w = this.width;
            const h = this.height; 
            const canvas = new OffscreenCanvas(w, h);
            const context = canvas.getContext("2d");
            const c = d3.color(this.label_color(label, index))
            row.forEach((pixel, i) => {
                c.opacity = s(pixel);
                context.fillStyle = c;
                const x = i % w;
                const y = Math.floor(i / w)
                context.fillRect(x, y, 1, 1)
            })
            return canvas;
        },
        draw_pcp_image: function(label, row, index, y, xs) {
            const dpr = window.devicePixelRatio || 1
            const w = this.width;
            const h = this.height;
            const N = this.dimensions.length;
            
            //return values.map((row, index) => {
            const canvas = new OffscreenCanvas(w, h);
            const context = canvas.getContext("2d");
            context.setTransform(dpr, 0,0,dpr,0,0)
            if (this.orientation == "horizontal") {
                context.rotate(-Math.PI / 2)
                context.translate(-w, 0)
            }
            context.strokeStyle = this.label_color(label, index);
            context.lineWidth = 1 / dpr;
            context.beginPath();

            context.moveTo(xs[0](row[0]) / dpr, y(0) / dpr)
            for (let i = 1; i < N; ++i) {
                const x = xs[i];
                context.lineTo(x(row[i]) / dpr, y(i) / dpr)
            }
            context.stroke();
            return canvas;
        },
        draw_radial_image: function(label, row, index, angle, radii) {
            const dpr = window.devicePixelRatio || 1
            const w = this.width / 2;
            const h = this.height / 2;
            const N = this.dimensions.length;
            
            const canvas = new OffscreenCanvas(2 * w, 2 * h);
            const context = canvas.getContext("2d");
            context.setTransform(dpr, 0, 0, dpr, 0, 0)
            context.translate(w / dpr, h / dpr)
            const c = d3.color(this.label_color(label, index));
            context.strokeStyle = c;
            context.lineWidth = .5 / dpr;

            for (let i = 0; i < N; ++i) {
                context.beginPath();
                context.moveTo(0, 0)
                const a = angle(i);
                const r = radii[i](row[i]);
                const x = Math.cos(a) * r / dpr;
                const y = Math.sin(a) * r / dpr;
                context.lineTo(x, y);
                context.closePath();
                context.stroke();
            }

            context.beginPath();
            const a0 = angle(0) / dpr;
            const r0 = radii[0](row[0]) / dpr;
            context.moveTo(Math.cos(a0) * r0, Math.sin(a0) * r0);
            for (let i = 1; i < N; ++i) {
                const a = angle(i);
                const r = radii[i](row[i]);
                const x = Math.cos(a) * r / dpr;
                const y = Math.sin(a) * r / dpr;
                context.lineTo(x, y);
            }
            /* context.stroke();
            context.fillStyle = c.copy({opacity: .5});
            context.fill(); */
            /* const radar = d3.areaRadial()
                .angle((d, i) => angle(i))
                .radius((d, i) => radii[i](d) / 2)
                //.innerRadius(() => 0)
                .curve(d3.curveCatmullRomClosed)
                .context(context);
            context.beginPath();
            radar(row);
            context.closePath(); */
            const c_fill = c.copy({opacity: .1});
            context.fillStyle = c_fill;
            context.fill();
            context.stroke();
            //c.opacity = .8;
            return canvas;

        },
        create_images: async function(values, labels) {
            const images = [];
            const dpr = 1;
            if (this.pointtype == "image") {
                const w = this.width;
                const h = this.height; 
                const s = d3.scaleLinear()
                    .domain(d3.extent(values.flat()))
                    .range([0, 1])
                for (let i = 0, n = labels.length; i < n; ++i) {
                    images.push(this.draw_mnist_image(labels[i], values[i], i, s));
                }
            } else if (this.pointtype == "parallel coordinates" && this.orientation != "radial") {
                const w = this.width;
                const h = this.height;
                const N = this.dimensions.length;
                const y = d3.scaleLinear()
                    .domain([0, N])
                    .range([0, h]);
                const xs = this.dimensions.map((_, i) => {
                    return d3.scaleLinear()
                        .domain(d3.extent(values, d => d[i]))
                        .range([1, w - 1]);
                });
                for (let i = 0, n = labels.length; i < n; ++i) {
                    images.push(this.draw_pcp_image(labels[i], values[i], i, y, xs));
                }
            } else if (this.pointtype == "parallel coordinates" && this.orientation == "radial") {
                const w = this.width;
                const h = this.height;
                const N = this.dimensions.length;
                const angle = d3.scaleLinear()
                    .domain([0, N])
                    .range([0, 2 * Math.PI]);
                const radii = this.dimensions.map((_, i) => {
                    return d3.scaleLinear()
                        .domain(d3.extent(values, d => d[i]))
                        .range([1, w - 1]);
                });
                for (let i = 0, n = labels.length; i < n; ++i) {
                    images.push(this.draw_radial_image(labels[i], values[i], i, angle, radii));
                }
            } else {
                return null;
            }
            return images;
        }
    }
    
}
</script>