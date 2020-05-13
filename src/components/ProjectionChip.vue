<template>
    <v-tooltip
        v-model="menu"
        top        
        transition="scale-transition"
        >
        <template v-slot:activator="{ on }">
            <v-chip
                outlined
                pill
                v-on="on">
                <v-avatar left :color="projection.status ? 'primary' : ''">
                    {{ projection.status }}
                </v-avatar>
                {{ title}}
            </v-chip>
        </template>

        <v-card width="18em">
            <!-- <v-list dense>
                <v-list-item v-if="metric">
                    <v-list-item-content>
                        Metric {{ metric }}
                    </v-list-item-content>
                </v-list-item>
                <v-list-item v-if="seed">
                    <v-list-item-content>
                        Seed {{ seed }}
                    </v-list-item-content>
                </v-list-item>
            </v-list> -->
            <v-simple-table dense>
                <tbody>
                    <tr>
                        <td>name</td>
                        <td>{{ projection.name }}</td>
                    </tr>
                    <tr v-for="(parameter, i) of projection.parameters.filter(d => d.value)" :key="i">
                        <td>{{ parameter.parameter }}</td>
                        <td>{{ parameter.value }}</td>
                    </tr>
                    <tr v-if="projection.seed">
                        <td>seed</td>
                        <td>{{ projection.seed }}</td>
                    </tr>
                    <tr v-if="projection.metric">
                        <td>metric</td>
                        <td>{{ projection.metric }}</td>
                    </tr>
                </tbody>
            </v-simple-table>
        </v-card>
    </v-tooltip>
</template>

<script>
export default {
    name: "projection-chip",
    props: ["projection"],
    data: function() {
        return {
            menu: false,
        }
    },
    computed: {
        title: function() {
            const projection = this.projection
            return `${projection.name} ${ projection.parameters.filter(d => d.value).map(d => d.value).join(" | ")}`
        },
        metric: function() {
            const metric = this.projection.parameters.find(d => d.metric)
            return metric ? metric.metric : null;
        },
        seed: function() {
            const seed = this.projection.parameters.find(d => d.seed)
            return seed ? seed.seed : null;
        },
    }

}
</script>
