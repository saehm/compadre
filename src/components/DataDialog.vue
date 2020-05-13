<template>
    <v-dialog 
        v-model="dialog" 
        max-width="600px">
        <template v-slot:activator="{ on }">
            <v-btn 
                icon
                v-on="on"
                :disabled="loading != null" >
                <v-tooltip bottom>
                    <template v-slot:activator="{ on }">
                        <v-icon v-on="on">mdi-folder</v-icon>
                    </template>
                    <span>data</span>
                </v-tooltip>
            </v-btn>
        </template>
        <v-card>
            <v-card-title primary-title>Data</v-card-title>
            <v-divider />
            <v-card-text>
                <v-select
                    v-model="data"
                    label="Data"
                    :items="datasets"
                    :hint="data.description"
                    persistent-hint
                    item-text="name"
                    prepend-icon="mdi-file-table"
                    return-object
                    >
                    <template v-slot:append-item>
                        <v-divider />
                        <v-file-input 
                            label="Use own dataset"
                            accept=".csv"
                            single-line
                            ref="file_input"
                            @change="file_input"
                            />
                    </template>
                </v-select>
                <v-switch
                    v-model="sparse"
                    label="Sparse dataset">

                </v-switch>
                
                <!-- <v-select
                    v-model="data.metric"
                    label="Metric"
                    :items="[
                        'euclidean',
                        'manhattan',
                        'cosine',
                        'chebyshev'
                    ]"
                    prepend-icon="mdi-ruler-square-compass">

                </v-select> -->

                <!-- <v-text-field
                    v-model="data.seed"
                    label="Seed"
                    prepend-icon="mdi-seed">

                </v-text-field> -->
            </v-card-text>
            <v-divider />
            <v-card-actions>
                <v-spacer />
                <v-btn text @click="dialog = false">Cancel</v-btn>
                <v-btn 
                    depressed 
                    @click="confirm_dialog" 
                    color="primary">
                    Load
                </v-btn>
            </v-card-actions>

        </v-card>
    </v-dialog>
</template>

<script>
import { mapState, mapActions } from "vuex";


export default {
    name: "data-dialog",
    data: function() {
        return {
            dialog: false,
        }
    },
    computed: {
        ...mapState({
            datasets: state => state.datasets.datasets,
            //name: state => state.data.name ? state.data.name : "",
        }),
        data: {
            get: function() {
                return this.$store.state.data;
            },
            set: function(index) {
                this.$store.commit("set_data", index);
            } 
        },
        sparse: {
            get: function() {
                return this.$store.state.data.sparse;
            },
            set: function(v) {
                this.$store.commit("set_sparse", v);
            }
        },
        loading: {
            get: function() {
                return this.$store.state.loading;
            },
            set: function(value) {
                this.$store.commit("set_loading", value);
            }
        },
    },
    methods: {
        ...mapActions([
            "add_to_datasets",
            "load_data",
        ]),
        file_input(file) {
            this.add_to_datasets(file);
            this.$refs.file_input.value = null;
        },
        async confirm_dialog() {
            this.dialog = false
            this.$store.commit("set_first_projection", null);
            await this.load_data();
        },
    }
}
</script>