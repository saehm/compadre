<template>
    <v-dialog 
        scrollable
        v-model="selection_dialog">
        <template v-slot:activator="{ on }">
            <v-btn
                icon
                title="Inspect Selection"
                v-on="on">
                <v-icon>mdi-selection-search</v-icon>
            </v-btn>
        </template>
            <v-card elevation="4">
                <v-card-title>
                    Selection
                    <v-spacer />
                    <v-text-field
                        v-model="selection_dialog_search"
                        append-icon="mdi-magnify"
                        label="Search"
                        single-line
                        hide-details>
                    </v-text-field>
                </v-card-title>
                <v-divider class="primary" />
                <v-card-text class="px-0">
                <v-lazy>
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
                </v-lazy>
                </v-card-text>
                <v-card-actions>
                    <v-spacer />
                    <v-btn color="primary" @click="selection_dialog = false">Close</v-btn>
                </v-card-actions>
            </v-card>
    </v-dialog>  
</template>

<script>
export default {
    name: "inspect-dialog",
    props: ["selection"],
    data: function() {
        return {
            selection_dialog: false,
            selection_dialog_search: "",
        }
    },
    computed: {
        subspaces: function() {
            return this.$store.state.subspaces;
        },
        label_color: function() {
            return this.$store.state.label_color;
        },
        dialog_table_headers: function() {
            const subspaces = this.subspaces;
            const headers = [
                {value: "key"}, 
                {text: "Name", value: "name", align: "left"},
                {text: "Class", value: "class"},
            ];

            if (this.$store.state.data.sparse) {
                (subspaces.length > 1 ? subspaces.slice(1) : subspaces).forEach(s => headers.push({
                    "text": s.name, 
                    "value": s.name.toLowerCase()
                }));
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
            const subspaces = this.subspaces;
            const sel = this.selection;
            if (sel) {
                sel.forEach((d, i) => {
                    if (d) {
                        const element = {
                            "name": points[i], 
                            "class": labels[i], 
                            "key": i
                        }
                        if (this.$store.state.data.sparse) {
                            (subspaces.length > 1 ? subspaces.slice(1) : subspaces).forEach(s => {
                                element[s.name.toLowerCase()] = values[i]
                                    .map((d, i) => [d, i])
                                    .filter((d, i) => s.columns[i])
                                    .filter(d => d[0] != 0)
                                    .map(d => dimensions[d[1]])
                                    .join(", ")
                            })
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
    }

}
</script>

<style scoped>

</style>