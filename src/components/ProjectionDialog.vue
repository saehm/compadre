<template>
   <!--  <v-dialog 
        v-model="dialog" 
        max-width="600px"> -->
        <!-- <template v-slot:activator="{ on }">
            <v-list-item>
            <v-list-item-content>
                Projections
            </v-list-item-content>
            <v-list-item-action>
                <v-btn icon v-on="on"><v-icon>mdi-plus-circle</v-icon></v-btn>
            </v-list-item-action>
        </v-list-item>
        </template> -->
        
        <v-card>
            <v-toolbar class="mb-2">
                <v-toolbar-title>
                    <slot></slot>
                </v-toolbar-title>
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
            <v-card-text>
                <v-tabs-items 
                    v-model="tab"
                    class="mb-4">
                    <v-tab-item 
                        v-for="(projection_method, i) of projection_methods" 
                        :key=i>
                        <v-row 
                            v-for="(parameter, j) of projection_method.parameters"
                            :key=j>
                            <v-range-slider
                                v-if="parameter.type === 'number'"
                                v-model="parameter.selection"
                                :min="parameter.min"
                                :max="parameter.max"
                                :step="parameter.step"
                                :label="parameter.name"
                                thumb-label>

                                <template v-slot:append>
                                    <v-text-field 
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

                            <v-select
                                v-if="parameter.type === 'metric'"
                                v-model="parameter.selection"
                                :items='["euclidean", "manhattan", "chebyshev", "jaccard"]'
                                label="Metric"
                                prepend-icon="mdi-ruler-square-compass">

                            </v-select>

                            <v-text-field
                                v-if="parameter.type === 'seed'"
                                v-model="parameter.selection"
                                label="Seed"
                                prepend-icon="mdi-sprout">

                            </v-text-field>
                        </v-row>
                    </v-tab-item>
                </v-tabs-items>
                <v-row justify="center">
                    <v-btn 
                        rounded
                        @click="list.push(...projection_list)">
                        <v-icon>
                            mdi-arrow-down
                        </v-icon> 
                        {{ parameter_list.length }}
                    </v-btn>
                </v-row>
            </v-card-text>
            <v-divider />
            <v-card-text>
                <v-subheader>
                    Compute
                </v-subheader>
                <v-sheet
                    v-if="list.length > 0"
                    class="mt-4"
                    min-height="100px">
                    <v-chip-group
                        column>
                        <v-chip 
                            v-for="(p, i) of list" 
                            :key=i
                            x-small>
                            {{ p.name }} {{ p.parameters.map(d => d.value).join(" | ")}}
                        </v-chip>
                    </v-chip-group>
                </v-sheet>
                <v-sheet
                    v-else>
                    empty
                </v-sheet>
            </v-card-text>
            <v-divider />
            <v-card-actions>
                <v-spacer />
                <v-btn text @click="dialog = false">Cancel</v-btn>
                <v-btn text @click="confirm_dialog(true)">Show</v-btn>
                <v-btn depressed @click="confirm_dialog(false)">
                    <v-badge class="align-self-center" right>
                        <template v-slot:badge>
                            {{ list.length }}
                        </template>
                        ADD
                    </v-badge>
                </v-btn>
            </v-card-actions>
        </v-card>
    </v-dialog>

</template>

<script>
import * as d3 from "d3";

export default {
    name: "projection-dialog",
    watch: {
        /* tab: function() {
            console.log(this.tab)
        } */
    },
    computed: {
        projection_list: function() {
            const parameter_list = this.parameter_list;
            return parameter_list.map(d => {
                return {
                    "name": this.projection_methods[this.tab].name,
                    "parameters": d,
                };
            })
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
                } else if (parameter.type === "metric") {
                    result.forEach(r => r.push({"metric": parameter.selection}))
                } else if (parameter.type === "seed") {
                    result.forEach(r => r.push({"seed": parameter.selection}))
                }
            }
            return result;
        }
    },
    data: function() {
        return {
            dialog: false,
            tab: 0,
            list: [],
            projection_methods: [
                {
                    "name": "tSNE",
                    "parameters": [
                        {
                            "name": "perplexity",
                            "type": "number",
                            "min": 10,
                            "max": 100,
                            "step": 1,
                            "selection": [50, 52],
                        },
                        {
                            "name": "epsilon",
                            "type": "number",
                            "min": 1,
                            "max": 20,
                            "step": 1,
                            "selection": [10, 10],
                        },
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                },
                {
                    "name": "TriMap",
                    "parameters": [
                        {
                            "name": "weight_adj",
                            "type": "number",
                            "min": 100,
                            "max": 10000,
                            "step": 100,
                            "selection": [500, 500],
                        },
                        {
                            "name": "c",
                            "type": "number",
                            "min": 1,
                            "max": 10,
                            "std": 5,
                            "step": 1,
                            "selection": [5, 5],
                        },
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                },
                {
                    "name": "UMAP",
                    "parameters": [
                        {
                            "name": "local_connectivity",
                            "type": "number",
                            "min": 1,
                            "max": 10,
                            "step": 1,
                            "selection": [1, 3],
                        },
                        {
                            "name": "min_dist",
                            "type": "number",
                            "min": 0,
                            "max": 2,
                            "step": .1,
                            "selection": [.5, .8],
                        },
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                },
                {
                    "name": "MDS",
                    "parameters": [
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }],
                },
                {
                    "name": "PCA",
                    "parameters": [
                        {
                            "type": "seed",
                            "selection": 1212,
                        }],
                },
                {
                    "name": "ISOMAP",
                    "parameters": [
                        {
                            "name": "neighbors",
                            "type": "number",
                            "min": 2,
                            "max": 100,
                            "step": 1,
                            "selection": [15, 17],
                        },
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                },
                {
                    "name": "LLE",
                    "parameters": [
                        {
                            "name": "neighbors",
                            "type": "number",
                            "min": 2,
                            "max": 100,
                            "step": 1,
                            "selection": [15, 17],
                        },
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                },
                {
                    "name": "LTSA",
                    "parameters": [
                        {
                            "name": "neighbors",
                            "type": "number",
                            "min": 2,
                            "max": 100,
                            "step": 1,
                            "selection": [15, 17],
                        },
                        {
                            "type": "metric",
                            "selection": "euclidean",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                },
                {
                    "name": "LDA",
                    "parameters": [
                        {
                            "name": "labels",
                            "type": "array",
                        },
                        {
                            "type": "seed",
                            "selection": 1212,
                        }
                    ],
                }

            ]
        }
    },
    methods: {
        confirm_dialog(showing) {
            console.log(showing, this.list)
            this.$store.dispatch("send_to_worker", this.list)
            this.dialog = false;
        }
    }
}
</script>