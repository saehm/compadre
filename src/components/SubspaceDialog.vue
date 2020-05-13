<template>
    <v-dialog 
        scrollable
        v-model="dialog" 
        max-width="600px">
        <template v-slot:activator="{ on }">
                <v-btn 
                    v-on=on 
                    rounded 
                    class="mx-auto my-4" 
                    color="primary">
                    <v-icon left>mdi-plus-circle</v-icon> 
                    create Subspace
                </v-btn>
            </v-row>
        </template>
        <v-card>
            <v-card-title>
                Create Subspace
            </v-card-title>
            <v-divider />
            <v-card-text>
                <v-form v-model="valid">
                    <v-text-field
                        label="Name"
                        :rules="name_rules"
                        v-model="name">
                    </v-text-field>
                    <v-combobox
                        label="Columns"
                        v-model="select"
                        :items="dimension_list"
                        :item-value="'index'"
                        :item-text="'name'"
                        multiple>

                    </v-combobox>
                    <v-text-field
                        v-model="fast_value"
                        label="Columns"
                        append-outer-icon="mdi-check"
                        :rules="fast_rules"
                        @click:append-outer="do_select">
                    </v-text-field>
                </v-form>
            </v-card-text>
            <v-divider />
            <v-card-actions>
                <v-spacer />
                <v-btn text @click="dialog=false">Cancel</v-btn>
                <v-btn 
                    depressed 
                    color="primary" 
                    :disabled="!valid"
                    @click="create_subspace">
                    OK
                </v-btn>
            </v-card-actions>
        </v-card>
    </v-dialog>
</template>

<script>
export default {
    name: "subspace-dialog",
    data: function() {
        return {
            dialog: false,
            name: "",
            cols: "",
            select: [],
            name_rules: [
                value => value.length > 3 || 'More than 3 characters',
            ],
            col_rules: [
                value => /[0-9]*/.test(value) || "Numbers, Comas and Minus allowed!"
            ],
            fast_rules: [
                value => /^(\d+(\-\d+)?\,)*$/m.test(value) || "Numbers, Comas and Minus allowed! (Close with a coma)",
            ],
            fast_value: "",
            valid: false,
        }
    },
    computed: {
        dimension_list: function() {
            return this.$store.state.dimensions.map((d, i) => {
                return {
                    'name': d, 
                    'index': i
                }
            });
        }
    },
    methods: {
        create_subspace: function() {
            const name = this.name;
            const cols = this.cols;
            console.log(name, cols)
            this.$store.commit("create_subspace", [name, this.select]);
            this.cols = "";
            this.name = "";
            this.select = [];
            this.dialog = false;
        },
        do_select: function() {
            const select = this.select;
            console.log("do_select", select)
            const values = this.fast_value.split(",").filter(v => v.length > 0);
            //const dimensions = this.$store.state.dimensions;
            const dimension_list = this.dimension_list;
            /* values.forEach(value => {
                let [from, to] = value.split("-");
                to = Math.min(to, dimensions.length - 1);
                from = Math.max(from, 0);
                if (to) {
                    for (let i = from; i < to; ++i) {
                        if (select.findIndex(d => d.index === i) < 0) {
                            select.push(dimensions[i])
                        }
                    }
                } else {
                    if (select.findIndex(d => d.index === from) < 0) {
                        select.push(dimensions[from])
                    }
                }
            }) */
            values.forEach(value => {
                let [from, to] = value.split("-");
                to = Math.min(to, dimension_list.length - 1);
                from = Math.max(from, 0);
                if (to) {
                    for (let i = from; i < to; ++i) {
                        if (select.findIndex(d => d.index === i) < 0) {
                            select.push(dimension_list[i])
                        }
                    }
                } else {
                    if (select.findIndex(d => d.index === from) < 0) {
                        select.push(dimension_list[from])
                    }
                }
            })
            this.fast_value = "";
        }
    },

}
</script>