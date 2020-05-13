<template>
    <v-col cols=12>
        <v-card tile>
            <v-toolbar
                dense 
                :collapse="!collapsed"
                min-width="20rem">
                <!-- <v-text-field
                    full-width
                    
                    filled
                    hide-details
                    single-line
                    dense        
                    prepend-icon="mdi-label"            
                    v-model="subspace.name">

                </v-text-field> -->
                <span class="title">
                    {{ subspace.name }}
                </span>
                <v-spacer />
                <v-btn icon @click="collapsed = !collapsed"><v-icon>{{ collapsed ? 'mdi-chevron-up' : 'mdi-chevron-down ' }}</v-icon></v-btn>
                <template slot="extension">
                    <v-tabs
                        align-with-title
                        show-arrows
                        v-model="tab">
                        <v-tab
                            v-for="(projection_method, i) of projection_methods" 
                            :key=i>
                            {{ projection_method.name }}
                        </v-tab>
                    </v-tabs>
                </template>
            </v-toolbar>
            <v-card-text v-if="collapsed">
                <v-tabs-items
                    v-model="tab">
                    <v-tab-item 
                        v-for="(projection_method, i) of projection_methods" 
                        :key=i>
                        <v-row dense>
                            <v-col
                                md=6
                                sm=12
                                xs=12
                                v-for="(parameter, j) of projection_method.parameters"
                                :key=j>
                                <v-range-slider 
                                    dense
                                    v-if="parameter.type === 'number'"
                                    v-model="parameter.selection"
                                    :min="parameter.min"
                                    :max="parameter.max"
                                    :step="parameter.step"
                                    :label="parameter.name"
                                    thumb-label>

                                    <template v-slot:append>
                                        <v-text-field 
                                            dense
                                            v-model="parameter.selection[1]" 
                                            class="mt-0 pt-0" 
                                            hide-details 
                                            single-line 
                                            :min="parameter.min"
                                            :max="parameter.max"
                                            :step="parameter.step"
                                            type="number" 
                                            style="width: 60px">
                                        </v-text-field>
                                    </template>
                                    <template v-slot:prepend>
                                        <v-text-field 
                                            dense
                                            v-model="parameter.selection[0]" 
                                            class="mt-0 pt-0" 
                                            hide-details 
                                            single-line 
                                            :min="parameter.min"
                                            :max="parameter.max"
                                            :step="parameter.step"
                                            type="number" 
                                            style="width: 60px">
                                        </v-text-field>
                                    </template>
                                </v-range-slider>

                            </v-col>
                            <v-col 
                                cols="6"
                                v-if="projection_method.seed">
                                <v-text-field 
                                    dense
                                    v-model="projection_method.seed"
                                    label="Seed"
                                    prepend-icon="mdi-sprout">

                                </v-text-field>
                            </v-col>
                            <v-col 
                                cols="6"
                                v-if="projection_method.metric">
                                <v-select 
                                    dense
                                    v-model="projection_method.metric"
                                    :items='["euclidean", "manhattan", "chebyshev", "cosine"]'
                                    label="Metric"
                                    prepend-icon="mdi-ruler-square-compass">

                                </v-select>
                            </v-col>
                        </v-row>
                    </v-tab-item>
                </v-tabs-items>
                <v-row dense justify="center">
                    <v-chip 
                        :disabled="projection_list.length <= 0"
                        color="primary"
                        @click="add_to_list">
                        <v-icon left>
                            mdi-plus
                        </v-icon> 
                        add {{ projection_list.length }} projections
                    </v-chip>
                </v-row>
            </v-card-text>
            <v-card-text>
                <v-row style="max-height: 8rem; overflow-y: auto">
                    <v-chip-group 
                        multiple 
                        column 
                        v-model="selection">
                        <projection-chip 
                            v-for="(projection, i) of projections" 
                            :key=i 
                            :projection="projection"
                            @click="projection.visible = true">
                        </projection-chip>
                    </v-chip-group>
                </v-row>
            </v-card-text>
            <v-divider v-if="selection.length > 0"  />
            <v-row>
                <v-col 
                    md=12 
                    sm=12 
                    v-for="(projection, i) of selection.map(j => projections[j])" 
                    :key="i">
                    <v-card-text>
                        <projection-row :projection="projection" />
                        <v-divider v-if="i < selection.length - 1" />
                    </v-card-text>
                </v-col>
            </v-row>
        </v-card>
    </v-col>
</template>

<script>
import * as d3 from "d3"
import ProjectionDialog from "./ProjectionDialog";
import ProjectionChip from "./ProjectionChip";
import ProjectionRow from "./ProjectionRow";

export default {
    name: "subspace-section",
    components: {
        ProjectionDialog,
        ProjectionChip,
        ProjectionRow,
    },
    props: ["subspace"],
    data: function() {
        return {
            tab: 0,
            /* list: [], */
            collapsed: true,
            selection: [],
            search_text: "",
        }
    },/* 
    watch: {
        selection: function() {
            console.log(this.selection)
        }
    }, */
    computed: {
        projections: function() {
            const subspace = this.subspace;
            const projections = this.$store.state.projections;
            return projections.filter(d => d.subspaceId == subspace.id);
        },
        projection_methods: function() {
            return this.$store.state.projection_methods.methods
        },
        projection_list: function() {
            const parameter_list = this.parameter_list;
            const list = this.projections;
            return parameter_list.map(d => {
                return {
                    "subspaceId": this.subspace.id,
                    "name": this.projection_methods[this.tab].name,
                    "parameters": d,
                    "seed": this.projection_methods[this.tab].seed,
                    "metric": this.projection_methods[this.tab].metric,
                };
            }).filter(new_element => {
                const l_i = list.findIndex(old_element => {
                    return JSON.stringify({
                        "metric": new_element.metric,
                        "name": new_element.name,
                        "parameters": new_element.parameters,
                        "seed": new_element.seed,
                        "subspaceId": new_element.subspaceId,
                    }) == JSON.stringify({
                        "metric": old_element.metric,
                        "name": old_element.name,
                        "parameters": old_element.parameters,
                        "seed": old_element.seed,
                        "subspaceId": old_element.subspaceId,
                    })
                })
                return l_i < 0
            });
        },
        parameter_list: function() {
            //if (this.tab) return 0;
            const projection = this.projection_methods[this.tab]
            let result = [[]];
            const parameters = projection.parameters//.filter(d => d.type != "metric" && d.type != "seed");
            for (let i = 0; i < parameters.length; ++i) {
                const parameter = parameters[i];
                if (parameter.type === "number") {
                    const [min, max] = parameter.selection;
                    const step = parameter.step;
                    const parameter_range = d3
                        .range(Math.max((max - min + step) / step, 1))
                        .map(d => min + d * step)
                        .map(d => {
                            return {
                                "parameter": parameter.name,
                                "value": d,
                            }
                        });
                    result = result.map(d => {
                        return parameter_range.map(p => [p, ...d]);
                    }).reduce((a, b) => [...a, ...b], []);
                }
            }
            return result;
        }
    },
    methods: {
        add_to_list: function() {
            //const list = this.list;
            const projection_list = this.projection_list;
            //const pool = this.$store.state.pool;
            const subspace = this.subspace;
            //console.log(subspace)
            this.$store.dispatch("send_to_worker", [projection_list, subspace]);
            //list.push(...projection_list);
        },
    },
    mounted() {
        //console.log(this.$store.state, this.$store.state.projection_methods.methods)
    }

}
</script>