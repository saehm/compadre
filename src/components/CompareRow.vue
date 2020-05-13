<template>
    <v-card class="mb-4">
        <v-toolbar 
            dense
            flat
            :color="local_selection ? 'black' : null"
            :dark="!!local_selection">
            <v-toolbar-title>
                {{ subspaces.find(s => s.id == left_projection.subspaceId).name }}:
                {{ left_projection.name }} 
                {{ left_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
                vs. 
                {{ subspaces.find(s => s.id == right_projection.subspaceId).name }}:
                {{ right_projection.name }} 
                {{ right_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
            </v-toolbar-title>
            <v-spacer />
            <v-toolbar-items>
                <!-- <v-btn
                    v-if="is_selected"
                    icon 
                    @click="selection = selection.map(() => true)">
                    <v-icon>mdi-select-off</v-icon>
                </v-btn> -->
                <inspect-dialog
                    :selection="local_selection"
                    v-if="!!local_selection" />
                <!-- <v-dialog 
                    v-model="selection_dialog" 
                    v-if="!!local_selection">
                    <template v-slot:activator="{ on }">
                        <v-btn
                            icon
                            title="Inspect Selection"
                            v-on="on">
                            <v-icon>mdi-selection-search</v-icon>
                        </v-btn>
                    </template>
                    <v-lazy><v-card>
                        <v-card-title>
                            <v-spacer />
                            <v-text-field
                                v-model="selection_dialog_search"
                                append-icon="mdi-magnify"
                                label="Search"
                                single-line
                                hide-details>
                            </v-text-field>
                        </v-card-title>
                        <v-card-text class="px-0">
                            <v-data-table
                                dense
                                :headers='dialog_table_headers'
                                :items="dialog_table_items"
                                :search="selection_dialog_search"
                                :items-per-page=5
                                item-key="key">
                                <template v-slot:item.class="{ item }">
                                    <v-badge dot left inline bordered :color="label_color(item.class, item.key)">
                                        {{ item.class}}
                                    </v-badge>
                                </template>
                            </v-data-table>
                        </v-card-text>
                        <v-card-actions>
                            <v-spacer />
                            <v-btn text @click="selection_dialog = false">Close</v-btn>
                        </v-card-actions>
                    </v-card>
                    </v-lazy>
                </v-dialog>    -->           
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
                    :left_projection="left_projection"
                    :right_projection="right_projection"
                    @set_reordering="set_reordering">

                </sorter-list>
                <v-btn 
                    icon
                    title="Remove Row"
                    @click="delete_me"><v-icon>mdi-close</v-icon>
                </v-btn>
            </v-toolbar-items>
            
        </v-toolbar>
        <v-divider />
        <v-card-text>
            <v-row dense> 
                <v-col cols="4" class="d-flex align-end justify-center">
                    {{ subspaces.find(s => s.id == left_projection.subspaceId).name }}:
                    {{ left_projection.name }} 
                    {{ left_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
                </v-col>
                <v-col cols="4" class="d-flex align-end justify-center">
                    <div ref="detail_col" />
                </v-col>
                <v-col cols="4" class="d-flex align-end justify-center">
                    {{ subspaces.find(s => s.id == right_projection.subspaceId).name }}:
                    {{ right_projection.name }} 
                    {{ right_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
                </v-col>
            </v-row>
            <v-row dense>
                <v-col cols="4">
                    <div ref="left_col" />
                </v-col>
                <v-col cols="4">
                    <div ref="mid_col" />
                </v-col>
                <v-col cols="4">
                    <div ref="right_col" />
                </v-col>
            </v-row>
        </v-card-text>
    </v-card>
</template>

<script>
import { mapState, mapActions } from "vuex";
import * as d3 from "d3";

import Vue from 'vue';
import ProjectionCanvas from "./ProjectionCanvas.vue"
import MatrixCanvas from "./MatrixCanvas.vue"
import SorterList from  "./SorterList.vue";
import DetailCanvas from "./DetailCanvas.vue";
import InspectDialog from "./InspectDialog.vue";

export default {
    name: "compare-row",
    components: {
        ProjectionCanvas,
        MatrixCanvas,
        SorterList,
        DetailCanvas,
        InspectDialog,
    },
    props: ["left_projection", "right_projection"],
    computed: {
        selection: {
            get: function() {
                return this.$store.state.selection;
            },
            set: function(array) {
                this.$store.commit("change_selection", array)
            }
        },
        is_selected: function() {
            return !this.selection.reduce((a, b) => a && b)
        },
        dpr: function() {
            return window.devicePixelRatio;
        },
        width: function() {
            const width = parseInt(window.getComputedStyle(this.$refs.left_col, null).width)
            const pl = parseInt(window.getComputedStyle(this.$refs.left_col, null).paddingLeft)
            const pr = parseInt(window.getComputedStyle(this.$refs.left_col, null).paddingRight)
            return (width - pl - pr) - 5;
            
        },
        left_canvas: function() {
            return this.$refs.left_canvas;
        },
        mid_canvas: function() {
            return this.$refs.mid_canvas;
        },
        right_canvas: function() {
            return this.$refs.right_canvas;
        },
        olc: function() {
            return this.left_canvas.transferControlToOffscreen();
        },
        omc: function() {
            return this.mid_canvas.transferControlToOffscreen();
        },
        orc: function() { 
            return this.right_canvas.transferControlToOffscreen();
        },
        ...mapState({
            projections: state => state.projections,
            subspaces: state => state.subspaces,
        }),
        hover_point: function() {
            const state = this.$store.state;
            return state.hover ? state.points[state.hover] : "";
        },
        hover_label: function() {
            const state = this.$store.state;
            return state.hover ? state.labels[state.hover] : "";
        },
        label_color: function() {
            return this.$store.state.label_color;
        }
    },
    data: function() {
        return {
            reordering: null,
            local_selection: null,
            selection_dialog: false,
            selection_dialog_search: "",
        }
    },
    mounted: function() {
        const width = this.width;
        const dpr = this.dpr;

        let ProjectionCanvasClass = Vue.extend(ProjectionCanvas);
        this.left_projection_canvas = new ProjectionCanvasClass({
            parent: this,
            propsData: {
                "projection": this.left_projection,
                "width": this.width,
                "height": this.width,
            }
        });
        this.left_projection_canvas.$mount(this.$refs.left_col);
        this.left_projection_canvas.$on("selection_end", this.selection_end)
        this.left_projection_canvas.$on("brushing", this.brushing);

        this.right_projection_canvas = new ProjectionCanvasClass({
            parent: this,
            propsData: {
                "projection": this.right_projection,
                "width": this.width,
                "height": this.width,
            }
        });
        this.right_projection_canvas.$mount(this.$refs.right_col);
        this.right_projection_canvas.$on("selection_end", this.selection_end)
        this.right_projection_canvas.$on("brushing", this.brushing);

        let MatrixCanvasClass = Vue.extend(MatrixCanvas);
        this.matrix_canvas = new MatrixCanvasClass({
            parent: this,
            propsData: {
                "left_projection": this.left_projection,
                "right_projection": this.right_projection,
                "reordering": this.reordering,
                "width": this.width,
                "height": this.width,
            }
        });
        this.matrix_canvas.$mount(this.$refs.mid_col);
        this.matrix_canvas.$on("selection_end", this.selection_end)
        this.matrix_canvas.$on("brushing", this.brushing);
            

        let DetailCanvasClass = Vue.extend(DetailCanvas);
        this.detail_canvas = new DetailCanvasClass({
            parent: this,
            propsData: {
                "left_projection": this.left_projection,
                "right_projection": this.right_projection,
                "width": this.width,
                "height": 75,
                "horizontal": true,
            }
        });
        this.detail_canvas.$mount(this.$refs.detail_col);
    },
    methods: {
        visibility_changed: function() {
            const left_projection = this.left_projection;
            const right_projection = this.right_projection;
            const left_canvas = this.olc;
            const mid_canvas = this.omc;
            const right_canvas = this.orc;

            const width = this.width;

            this.$store.dispatch("draw_comparison", [left_projection, right_projection, left_canvas, mid_canvas, right_canvas, width]);

        },
        delete_me: function() {
            const left_projection = this.left_projection;
            const right_projection = this.right_projection;
            this.$store.dispatch("delete_comparison", [left_projection, right_projection]);
            //this.$destroy();
            this.left_projection_canvas.$destroy();
            this.right_projection_canvas.$destroy();
            this.matrix_canvas.$destroy()
            this.detail_canvas.$destroy();
        },
        set_reordering: function(reordering) {
            this.matrix_canvas.set_reordering(reordering.reordering)
        },
        selection_end: async function(new_selection) {
            if (new_selection && !new_selection.reduce((a,b) => a && b)) {
                this.local_selection = new_selection;
                this.left_projection_canvas.local_selection = new_selection;
                this.right_projection_canvas.local_selection = new_selection;
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
            
            this.right_projection_canvas.lasso_selection = null; 
            this.right_projection_canvas.local_selection = null; 
            this.right_projection_canvas.$refs.lasso.setAttribute('d', '')
            this.left_projection_canvas.lasso_selection = null; 
            this.left_projection_canvas.local_selection = null; 
            this.left_projection_canvas.$refs.lasso.setAttribute('d', '');
            
            this.matrix_canvas.local_selection = null;
            d3.select(this.matrix_canvas.$refs.svg).call(this.matrix_canvas.brush.move, null)

        },
        reject_selection: function() {
            this.selection = this.selection.map(() => true); 
            this.local_selection = null;
            
            this.right_projection_canvas.lasso_selection = null; 
            this.right_projection_canvas.local_selection = null; 
            this.right_projection_canvas.$refs.lasso.setAttribute('d', '');
            this.left_projection_canvas.lasso_selection = null; 
            this.left_projection_canvas.local_selection = null; 
            this.left_projection_canvas.$refs.lasso.setAttribute('d', '');
            
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
            this.right_projection_canvas.local_selection = sel;
            this.left_projection_canvas.local_selection = sel;
        },
    },
    beforeDestroy: function() {
        console.log("destroy comparerow")
        this.delete_me();
    }
}
</script>
