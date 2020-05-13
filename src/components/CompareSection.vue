<template>
     <v-col cols=12>
         <v-card tile>
            <v-toolbar
                dense 
                min-width="20rem">
                <v-toolbar-title>
                    Compare Projections
                </v-toolbar-title>
                <v-spacer />
                
                <!-- <v-btn icon><v-icon>mdi-dots-vertical</v-icon></v-btn>
                 -->
            </v-toolbar>
            <v-divider class="primary" />
        <v-card-text>
            <v-form v-model="valid" :lazy-validation=false>
                <v-row>
                    <v-col cols=6>
                        <v-select
                            v-model="left_projection"
                            label="Left Projection"
                            :rules="select_projections"
                            :items="projections"
                            :item-text="d => `${ d.status || '' } ${subspaces.find(s => s.id == d.subspaceId).name}: ${ d.name } ${ d.parameters.filter(d => d.value).map(d => d.value).join(' | ') }`"
                            :item-value="d => d">

                        </v-select>
                    </v-col>
                    <v-col cols=6>
                        <v-select
                            v-model="right_projection"
                            label="Right Projection"
                            :rules="select_projections"
                            :items="projections"
                            :item-text="d => `${ d.status || '' } ${subspaces.find(s => s.id == d.subspaceId).name}: ${ d.name } ${ d.parameters.filter(d => d.value).map(d => d.value).join(' | ') }`"
                            :item-value="d => d">

                        </v-select>
                    </v-col>
                </v-row>
            </v-form>
            <v-row justify="center">
                <v-btn 
                    rounded
                    :disabled="!valid"
                    @click="append_comparison">
                    <v-icon left>
                        mdi-plus
                    </v-icon> 
                    Comparison
                </v-btn>
            </v-row>
        </v-card-text>
        
        <v-card-text>
            <v-row dense>
            <v-col md="12" sm="12">
                <compare-row
                    v-for="(comparison, i) of comparisons"
                    :key=i
                    :left_projection=comparison.left_projection
                    :right_projection=comparison.right_projection
                    >
                </compare-row>
            </v-col>
        </v-row>
        </v-card-text>
     </v-card>
     </v-col>
</template>

<script>
import { mapState, mapActions } from "vuex";
import CompareRow from "./CompareRow";

export default {
    name: "compare-section",
    components: {
        CompareRow,
    },
    data: function() {
        return {
            left_projection: null,
            right_projection: null,
            valid: false,
            select_projections: [
                value => value != null || "select projection"
            ]
        }
    },
    computed: {
        ...mapState({
            projections: state => state.projections.sort((a, b) => b.status - a.status),
            subspaces: state => state.subspaces,
            comparisons: state => state.comparisons
        })
    },
    methods: {
        append_comparison: function() {
            //console.log(this.left_projection, this.right_projection)
            this.$store.state.comparisons.push(
                {"left_projection": this.left_projection, 
                "right_projection": this.right_projection})
        }
    }
}
</script>