<template>
    <v-menu offset-y>
        <template v-slot:activator="{ on }">
            <v-btn icon v-on="on" :loading="loading">
                <v-icon>mdi-sort</v-icon>
            </v-btn>
        </template>
        <v-list dense>
            <v-list-item-group>
                <v-list-item dense @click="reorder('none')">
                    reset ordering
                </v-list-item>
            </v-list-item-group>
            <v-subheader>
                SORT MATRIX
            </v-subheader>
            <v-list-item-group>
                <v-list-item @click="reorder('olo')">
                    Optimal Leaf Ordering
                </v-list-item>
                <!-- <v-list-item @click="reorder('bo')">
                    Barycenter Ordering
                </v-list-item> -->
                <v-list-item @click="reorder('so')">
                    Spectral Ordering
                </v-list-item>
            </v-list-item-group><v-subheader>
                SORT MATRIX WITHIN CLASS
            </v-subheader>
            <v-list-item-group>
                <v-list-item @click="reorder_within('wolo')">
                    Optimal Leaf Ordering (within class)
                </v-list-item>
                <!-- <v-list-item @click="reorder('bo')">
                    Barycenter Ordering
                </v-list-item> -->
                <v-list-item @click="reorder_within('wso')">
                    Spectral Ordering (within class)
                </v-list-item>
            </v-list-item-group>
            <v-subheader v-if="reorderings.length > 0">
                REORDER
            </v-subheader>
            <v-list-item-group v-if="reorderings.length > 0">
                <v-list-item two-line dense
                    v-for="(reorder, i) of reorderings" 
                    :key=i 
                    @click="$emit('set_reordering', reorder)"
                    >
                    <v-list-item-content>
                        <v-list-item-title v-if="!reorder.right_projection">
                            Projection {{ subspaces.find(s => s.id == reorder.left_projection.subspaceId).name }}:
                        {{ reorder.left_projection.name }} 
                        {{ reorder.left_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
                        </v-list-item-title>
                        <v-list-item-title v-else>
                            Comparison {{ subspaces.find(s => s.id == reorder.left_projection.subspaceId).name }}:
                        {{ reorder.left_projection.name }} 
                        {{ reorder.left_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
                        vs. 
                        {{ subspaces.find(s => s.id == reorder.right_projection.subspaceId).name }}:
                        {{ reorder.right_projection.name }} 
                        {{ reorder.right_projection.parameters.filter(d => d.value).map(d => d.value).join(' | ') }}
                        </v-list-item-title>
                        <v-list-item-subtitle>
                            {{ {
                                    "so": "Spectral Ordering", 
                                    "bo": "Barycenter Ordering",
                                    "olo": "Optimal Leaf Ordering",
                                    "wso": "Spectral Ordering (within class)", 
                                    "wbo": "Barycenter Ordering (within class)",
                                    "wolo": "Optimal Leaf Ordering (within class)",
                            }[reorder.type] }}
                        </v-list-item-subtitle>
                    </v-list-item-content>
                </v-list-item>
                
            </v-list-item-group>
        </v-list>
    </v-menu>
</template>

<script>
import { mapState } from "vuex";
import { spawn, Worker, Pool, Transfer } from "threads";

export default {
    name: "sorter-list",
    props: ["left_projection", "right_projection"],
    data: function() {
        return {
            loading: false,
        }
    },
    computed: {
        ...mapState({
            subspaces: state => state.subspaces,
            worker: state => state.workers[(state.workers.length - 2) % state.workers.length],
            reorderings: state => state.reorderings,
        }),
    },
    methods: {
        reorder: async function (reorder_type) {
            this.loading = true;
            const left_projection = this.left_projection;
            const right_projection = this.right_projection;
            const reordering = await this.$store.dispatch("reorder_matrix", [left_projection, right_projection, reorder_type])
            this.$emit("set_reordering", reordering);
            this.loading = false;
        },
        reorder_within: async function (reorder_type) {
            this.loading = true;
            const left_projection = this.left_projection;
            const right_projection = this.right_projection;
            const reordering = await this.$store.dispatch("reorder_matrix_within", [left_projection, right_projection, reorder_type])
            this.$emit("set_reordering", reordering);
            this.loading = false;
        },
    }
}
</script>