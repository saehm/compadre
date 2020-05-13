<template>
  <v-app>

     <div>

      <v-badge 
        class="px-2"
        dot 
        left 
        inline 
         
        :color="label_color('InfoVis')">
        {{ "Vis" }}
      </v-badge>

      <v-badge 
        class="px-2"
        dot 
        left 
        inline 
         
        :color="label_color('VAST')">
        {{ "VAST" }}
      </v-badge>

      <v-badge 
        class="px-2"
        dot 
        left 
        inline 
         
        :color="label_color('SciVis')">
        {{ "SciVis" }}
      </v-badge>

      <v-badge 
        class="px-2"
        dot 
        left 
        inline 
         
        :color="label_color('Vis')">
        {{ "InfoVis" }}
      </v-badge>
     </div>
    <v-tooltip
      :dark="!theme"
      top
      :position-x="tooltip ? tooltip.left : null"
      :position-y="tooltip ? tooltip.top : null"
      v-model=tooltip>
      {{ hovered_point }} 
      <v-badge 
        class="px-2"
        v-if="hovered_point" 
        dot 
        left 
        inline 
        bordered 
        :color="label_color(hovered_label, $store.state.hover)">
        {{ hovered_label }}
      </v-badge>
    </v-tooltip>

    <v-app-bar clipped-left dense app :color="!theme ? 'primary' : null">
      <v-app-bar-nav-icon 
        @click="drawer = !drawer" 
        :loading="deleteing || loading != null">
      </v-app-bar-nav-icon>
      <v-toolbar-title class="headline text-uppercase">
        <span>COMPA</span>
        <span class="font-weight-light">DRE</span>{{ name ? ': ' : ''}} {{ name }}
      </v-toolbar-title>
      <v-spacer />
      <v-toolbar-items>
        <data-dialog 
          @loading_data="loading_data"/>
        <appearance-dialog />
        <v-btn 
          :disabled="loading != null"
          icon 
          @click="theme = !theme">
          <v-icon>mdi-invert-colors</v-icon>
        </v-btn>
        
      </v-toolbar-items>
      <v-progress-linear 
        :active="loading != null"
        striped
        indeterminate
        query
        absolute bottom />
    </v-app-bar>

    <v-navigation-drawer
      v-model="drawer" 
      app 
      temporary
      clipped>
      <v-list>
        <v-list-item @click="delete_db">
          <v-list-item-icon><v-icon>mdi-database-remove</v-icon></v-list-item-icon>
          <v-list-item-content>empty db</v-list-item-content>
        </v-list-item>
        <v-list-item @click="save_db">
          <v-list-item-icon><v-icon>mdi-content-save</v-icon></v-list-item-icon>
          <v-list-item-content>export db</v-list-item-content>
        </v-list-item>
        <v-list-item @click="load_db">
          <v-list-item-icon><v-icon>mdi-upload</v-icon></v-list-item-icon>
          <v-list-item-content>import db</v-list-item-content>
        </v-list-item>
        <input ref="file_input" type="file" name="name" style="display: none;" />
      </v-list>
    </v-navigation-drawer>

    
    <v-content v-if="loading">
      <v-row style="height: 50vh"
        class="fill-height"
        align-content="center"
        justify="center">
        <v-col 
          class="subtitle-1 text-center"
          cols=12>
          {{ loading }}
        </v-col>
        <v-col 
          cols=4>
          <v-progress-linear 
            query
            rounded
            indeterminate />
        </v-col>
      </v-row>
    </v-content>

    <v-content v-else>
      <v-row>
        
        <subspace-section 
          v-for="(subspace, i) of subspaces" 
          :key="i"
          :subspace="subspace" />
      </v-row>
      <v-row>
        <subspace-dialog />
      </v-row>
      <v-row>
        <compare-section />
      </v-row>
    </v-content>   

    <v-footer class="mt-12">
      
      <v-spacer />
      
    </v-footer>

    <v-bottom-sheet 
      v-if="threads && threads.length > 0"
      v-model="threads"
      hide-overlay
      inset
      scrollable 
      :dark="!theme"
      persistent>
      <v-sheet 
        style="overflow-y:scroll"
        class="text-center pa-1"
        height="3.5rem">
        <v-chip-group 
            column 
            v-model="threads">
            <v-chip  
              @click="cancel_all">
              <v-icon left>mdi-cancel</v-icon>
              cancel all {{ threads.length }} thread{{ threads.length > 1 ? "s" : "" }}
            </v-chip>
            <v-chip v-for="(t, i) of threads" :key=i 
              pill 
              outlined 
              close
              close-icon="mdi-close"
              @click:close="remove_thread(t.thread)">
                <v-avatar left>
                  <v-progress-circular indeterminate size=16 width=2 />
                </v-avatar>
                {{ t.projection.name }} {{ t.projection.parameters.filter(d => d.value).map(d => d.value).join(" | ")}}
            </v-chip>
        </v-chip-group>
      </v-sheet>
    </v-bottom-sheet>

  </v-app>
</template>

<script>
import { mapState } from "vuex";
import DataDialog from "./components/DataDialog";
import AppearanceDialog from "./components/AppearanceDialog";
import SubspaceSection from "./components/SubspaceSection";
import SubspaceDialog from "./components/SubspaceDialog";
import CompareSection from "./components/CompareSection";

export default {
  name: 'App',
  components: {
    DataDialog,
    AppearanceDialog,
    SubspaceSection,
    SubspaceDialog,
    CompareSection,
  },
  data: () => ({
    drawer: false,
    theme: true,
    /* loading: true, */
    deleteing: false,
  }),
  watch: {
    theme: function() {
      this.$vuetify.theme.dark = this.theme;
    },
    hovered_point: function() {
      const hover = this.hover;
      const points = this.points;
    },
  },
  computed: {
    ...mapState({
      datasets: state => state.datasets,
      subspaces: state => state.subspaces,
      threads: state => state.threads,
      points: state => state.points,
      hover: state => state.hover,
      hovered_point: state => state.hover ? state.points[state.hover] : "",
      hovered_label: state => state.hover ? state.labels[state.hover] : "",
      name: state => state.data_name ? state.data_name : "",
      label_color: state => state.label_color,
    }),
    loading: {
      get: function() {
        return this.$store.state.loading;
      },
      set: function(value) {
        this.$store.commit("set_loading", value);
      }
    },
    tooltip: {
        get: function() {
            return this.$store.state.tooltip;
        },
        set: function(msg) {
            this.$store.commit("set_tooltip", msg);
        }
    },
  },
  created() {
    this.$vuetify.theme.dark = this.theme;
    this.$store.dispatch("load_data")//.then(() => this.loading = false)
  },
  methods: {
    delete_db: async function() {
      if (window.confirm("actual cache get lost!")) {
        const threads = this.$store.state.threads.forEach(t => t.thread.cancel())
        this.deleteing = true;
        await this.$store.dispatch("delete_db")
        this.deleteing = false;
        location.reload();
      }
      this.drawer = false
    },
    remove_thread: function(thread) {
      const state = this.$store.state;
      const t_i = state.threads.findIndex(d => d.thread == thread);
      state.threads.splice(t_i, 1);
    },
    cancel_all: function() {
      const state = this.$store.state;
      state.threads.forEach(async t => await t.thread.cancel());
      state.threads = [];
    },
    loading_data: function(val) {
      //this.loading = val;
    },
    save_db: async function() {
      this.deleteing = true;
      await this.$store.dispatch("get_db");
      this.deleteing = false;
      
    },
    load_db: function() {
      const input = this.$refs.file_input;
      const store = this.$store;
      const f = (v) => {
        this.deleteing = v;
        this.loading = "load file...";
      }
      input.onchange = async e => {
        f(true)
        var file = e.target.files[0];
        await store.dispatch("load_db", file);
        f(false);
        location.reload();
      }
      if (window.confirm("actual cache get lost!")) input.click();
      this.drawer = false
    }
  },
};
</script>