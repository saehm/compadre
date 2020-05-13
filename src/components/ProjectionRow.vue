<template>
    <v-card>
        <v-toolbar 
            dense
            flat
            :class="!!local_selection ? 'black' : null"
            :dark="!!local_selection">
            <v-toolbar-title>
                {{ projection.name }} {{ projection.parameters.map(d => d.value).join(" | ")}}
            </v-toolbar-title>
            <v-rating
                class="mx-4"
                dense
                hover
                background-color="primary"
                v-model="projection.status"
                @input="change_status(projection.id, projection.status)">

            </v-rating>
            <v-spacer />
            <v-toolbar-items>
                <inspect-dialog 
                    :selection="local_selection" 
                    v-if="!!local_selection" />             
                <v-btn 
                    v-if="!!local_selection"
                    icon
                    title="Accept Selection"
                    @click="accept_selection">
                    <v-icon>mdi-check</v-icon>
                </v-btn>
                <v-btn
                    v-if="!selection.reduce((a, b) => a && b) || local_selection"
                    icon 
                    title="Reject Selection"
                    @click="reject_selection">
                    <v-icon>mdi-select-off</v-icon>
                </v-btn>
                <v-btn 
                    v-if="!!local_selection"
                    icon
                    title="Move Selection Top Left"
                    @click="manual_selection">
                    <v-icon>mdi-arrow-top-left</v-icon>
                </v-btn>
            
            <sorter-list
                :left_projection="projection"
                :right_projection="null"
                @set_reordering="set_reordering">

            </sorter-list>
            </v-toolbar-items>
            
        </v-toolbar>
        <v-card-text>
            <v-row ref="data_row" dense>
                <v-col cols="5">
                    <div ref="left_col" />
                </v-col>
                <v-col cols="5">
                    <div ref="right_col" />
                </v-col>
                <v-col cols="2">
                    <div ref="detail_col" />
                </v-col>
            </v-row>
        </v-card-text>
        <!-- <v-divider />
        <v-card-actions>
            {{ hover_point }}  <v-badge class="px-2" v-if="hover_point" dot left inline bordered :color="label_color(hover_label, $store.state.hover)">{{ hover_label }}</v-badge>
            <v-spacer />
            <v-rating
                class="px-2"
                    dense
                    hover
                    background-color="primary"
                    v-model="projection.status"
                    @input="change_status(projection.id, projection.status)">

                </v-rating>
        </v-card-actions> -->
    </v-card>
</template>

<script>
import Vue from 'vue';
import ProjectionCanvas from "./ProjectionCanvas.vue";
import MatrixCanvas from "./MatrixCanvas.vue";
import DetailCanvas from "./DetailCanvas.vue";
import SorterList from  "./SorterList.vue";
import InspectDialog from "./InspectDialog.vue";

import * as d3 from "d3";
import { mapState } from "vuex";
//import { spawn, Worker, Pool, Transfer } from "threads";

export default {
    name: "projection-row",
    components: {
        ProjectionCanvas,
        MatrixCanvas,
        SorterList,
        DetailCanvas,
        InspectDialog,
    },
    data: function() {
        return {
            local_selection: null,
            selection_dialog: false,
            selection_dialog_search: "",
        }
    },
    props: ["projection"],
    computed: {
        dialog_table_headers: function() {
            const headers = [
                {value: "key"}, 
                {text: "Name", value: "name", align: "left"},
                {text: "Class", value: "class"},
            ];

            if (this.$store.state.data.name == "VISPUBDATA") {
                headers.push({
                    text: "Coauthors", 
                    value: "coauthors",
                });
                headers.push({
                    text: "Keywords", 
                    value: "keywords",
                });
            } else {
                this.$store.state.dimensions.map(d => {
                    headers.push({"text": d, "value": d});
                })
            }
            return headers
        },
        dialog_table_items: function() {
            const table = [];
            const points = this.$store.state.points;
            const labels = this.$store.state.labels;
            const values = this.$store.state.values;
            const dimensions = this.$store.state.dimensions;
            const sel = this.local_selection;
            if (sel) {
                sel.forEach((d, i) => {
                    if (d) {
                        const element = {
                            "name": points[i], 
                            "class": labels[i], 
                            "key": i
                        }
                        if (this.$store.state.data.name == "VISPUBDATA") {
                            element["coauthors"] = values[i]
                                .slice(0, 610)
                                .map((d, i) => [d, i])
                                .filter(d => d[0] != 0)
                                .map(d => dimensions[d[1]])
                                .join(", ");

                            element["keywords"] = values[i].slice(611)
                                .map((d, i) => [d, i])
                                .filter(d => d[0] != 0)
                                .map(d => d[1] + 611)
                                .map(i => dimensions[i])
                                .join(", ");
                        } else {
                            this.$store.state.dimensions.forEach((d, j) => {
                                element[d] = values[i][j];
                            })
                        }
                        table.push(element);
                    }
                })
            }
            return table;
        },
        selection: {
            get: function() {
                return this.$store.state.selection;
            },
            set: function(array) {
                this.$store.commit("change_selection", array)
            }
        },
        ...mapState({
            worker: state => state.workers[0],
            label_color: state => state.label_color,
        }),
        dpr: function() {
            return window.devicePixelRatio;
        },
        width: function() {
            const width = parseInt(window.getComputedStyle(this.$refs.left_col, null).width);
            const pl = parseInt(window.getComputedStyle(this.$refs.left_col, null).paddingLeft);
            const pr = parseInt(window.getComputedStyle(this.$refs.left_col, null).paddingRight);
            return width - pl - pr;
        },
        detail_width: function() {
            const width = parseInt(window.getComputedStyle(this.$refs.detail_col, null).width);
            const pl = parseInt(window.getComputedStyle(this.$refs.detail_col, null).paddingLeft);
            const pr = parseInt(window.getComputedStyle(this.$refs.detail_col, null).paddingRight);
            return width - pl - pr;
        },
        detail_width_test: function() {
            const width = parseInt(window.getComputedStyle(this.$refs.detail_col_test, null).width);
            const pl = parseInt(window.getComputedStyle(this.$refs.detail_col_test, null).paddingLeft);
            const pr = parseInt(window.getComputedStyle(this.$refs.detail_col_test, null).paddingRight);
            return width - pl - pr;
        },/* 
        detail_height: function() {
            const height = parseInt(window.getComputedStyle(this.$refs.detail_col, null).height);
            const pt = parseInt(window.getComputedStyle(this.$refs.detail_col, null).paddingTop);
            const pb = parseInt(window.getComputedStyle(this.$refs.detail_col, null).paddingBottom);
            return height - pt// - pb;
        }, */
        hover_point: function() {
            const state = this.$store.state;
            return state.hover ? state.points[state.hover] : "";
        },
        hover_label: function() {
            const state = this.$store.state;
            return state.hover ? state.labels[state.hover] : "";
        },
    },
    mounted: async function() {
        await this.init();
    },
    methods: {
        init: async function() {
            let ProjectionCanvasClass = Vue.extend(ProjectionCanvas);
            this.projection_canvas = new ProjectionCanvasClass({
                parent: this,
                propsData: {
                    "projection": this.projection,
                    "width": this.width,
                    "height": this.width,
                    //"local_selection": this.local_selection,
                }
            });
            this.projection_canvas.$mount(this.$refs.left_col);
            this.projection_canvas.$on("selection_end", this.selection_end)
            this.projection_canvas.$on("brushing", this.brushing);

            let MatrixCanvasClass = Vue.extend(MatrixCanvas);
            this.matrix_canvas = new MatrixCanvasClass({
                parent: this,
                propsData: {
                    "left_projection": this.projection,
                    "right_projection": null,
                    "width": this.width,
                    "height": this.width,
                    //"reordering": null,
                    //"local_selection": this.local_selection,
                },
            });
            this.matrix_canvas.$mount(this.$refs.right_col);
            this.matrix_canvas.$on("selection_end", this.selection_end)
            this.matrix_canvas.$on("brushing", this.brushing);
            
            let DetailCanvasClass = Vue.extend(DetailCanvas);
            this.detail_canvas = new DetailCanvasClass({
                parent: this,
                propsData: {
                    "left_projection": this.projection,
                    "right_projection": null,
                    "width": this.detail_width,
                    "height": this.width,
                    "horizontal": false,
                }
            });
            
            this.detail_canvas.$mount(this.$refs.detail_col);
        },
        set_reordering: function(reordering) {
            this.matrix_canvas.set_reordering(reordering.reordering)
        },
        change_status: function(projectionId, status) {
            this.$store.commit('change_status', [projectionId, status]);
        },
        remove_me: function() {
            this.projection.visible = false;
            this.projection_canvas.$destroy();
            this.matrix_canvas.$destroy()
            this.detail_canvas.$destroy();
        },
        selection_end: async function(new_selection) {
            if (new_selection && !new_selection.reduce((a,b) => a && b)) {
                this.local_selection = new_selection;
                this.projection_canvas.local_selection = new_selection;
                this.matrix_canvas.local_selection = new_selection;
                
                /* this.matrix_canvas.local_selection = new_selection;
                this.projection_canvas.local_selection = new_selection; */
            } else {
                this.local_selection = null;
            }
        },
        accept_selection: function() {
            this.selection = this.local_selection; 
            this.local_selection = null; 
            
            this.projection_canvas.lasso_selection = null; 
            this.projection_canvas.local_selection = null; 
            this.projection_canvas.$refs.lasso.setAttribute('d', '')

            this.matrix_canvas.local_selection = null;
            d3.select(this.matrix_canvas.$refs.svg).call(this.matrix_canvas.brush.move, null)

        },
        reject_selection: function() {
            this.selection = this.selection.map(() => true); 
            this.local_selection = null;
            
            this.projection_canvas.lasso_selection = null; 
            this.projection_canvas.local_selection = null; 
            this.projection_canvas.$refs.lasso.setAttribute('d', '');
            
            this.matrix_canvas.local_selection = null;
            d3.select(this.matrix_canvas.$refs.svg).call(this.matrix_canvas.brush.move, null)

        },
        manual_selection: async function() {
            const matrix_canvas = this.matrix_canvas;
            const local_selection = this.local_selection;
            const ordered = matrix_canvas.reordering || local_selection
                .map((_, i) => i);
            const selected = ordered.filter((d, i) => local_selection[d] == true)
            const unselected = ordered.filter((d, i) => local_selection[d] != true)
            const index = [...selected, ...unselected];
            matrix_canvas.set_reordering(index);
            await matrix_canvas.draw();
            //this.reject_selection();
            await matrix_canvas.move_brush_top_left(selected);
        },
        brushing: function(sel) {
            console.log("brushing row", sel)
            this.local_selection = sel;
            this.projection_canvas.local_selection = sel;
        },
        resize: function(e) {
            /* console.log(e)
            if (!e) return;
            this.projection_canvas.width = this.width;
            this.matrix_canvas.width = this.width;
            this.detail_canvas.width = this.detail_width;
            this.projection_canvas.height = this.width;
            this.matrix_canvas.height = this.width;
            this.detail_canvas.height = this.width;
            this.projection_canvas.draw();
            this.matrix_canvas.draw();
            this.detail_canvas.draw(); */
        }
    },
    beforeDestroy: function() {
        console.log("destroy projectionrow")
        this.remove_me();
    }
}
</script>