var app = new Vue({
  el: '#app',
  delimiters: ["[[", "]]"],
  data: {
    database: '',
    database_name: '',
    database_list: [],
    current_id: '',
    page_start_index: 0,
    page_index: 0,
    row_count: 4,
    column_count: 4,
    page_count: 0,
    overview: [],
    page_overview: [],
    stats: {},
    detail: {},
    last_action: {},
    draw_bin: 1,
    suffix: 'ed-v',
    filter: {
      reward: -1,
      final_d_lower: 0,
      final_d_upper: 0.1,
      id: '',
    },
  },
  filters: {
    round: function(value, decimals) {
      if (!value) { value = 0; }
      if (!decimals) { decimals = 0; }
      return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }
  },
  methods: {
    loadDatabase: function(event) {
      app.database_name = app.database.split("data/")[1];
      localStorage.setItem('database', app.database);

      $.get("/api/overview", {database: app.database, reward: app.filter.reward, id: app.filter.id, final_d_lower: app.filter.final_d_lower, final_d_upper: app.filter.final_d_upper}, (data) => {
        app.overview = data.reverse();

        app.page_count = Math.ceil(app.overview.length / (app.row_count * app.column_count));
        app.updatePage();

        if (app.overview.length > 0) {
          app.current_id = app.overview[0].id;
          app.updateDetail();

          $.get("/api/action/" + app.current_id, {"database": app.database}, (data) => {
            app.last_action = app.objectifyGraspResult(data);
          });
        }
      });

      $.get("/api/stats", {database: app.database}, (data) => {
        app.stats = data;
      });
    },
    updateDetail: function(event) {
      $.get("/api/action/" + app.current_id, {"database": app.database}, (data) => {
        app.detail = app.objectifyGraspResult(data);
      });
    },
    updatePage: function(event) {
      localStorage.setItem('row_count', app.row_count);
      localStorage.setItem('column_count', app.column_count);
      localStorage.setItem('draw_bin', app.draw_bin);

      if (app.overview) {
        app.page_overview = app.overview.slice(app.page_start_index, app.page_start_index + app.row_count * app.column_count);
      }
    },
    diffIndex: function(index_diff) {
      index_diff = parseInt(index_diff);
      let index = app.overview.findIndex((e) => { return e.id == app.current_id; });
      if (0 <= index + index_diff && index + index_diff < app.overview.length) {
        index += index_diff;
        app.current_id = app.overview[index].id;

        if (index > app.page_start_index + app.row_count * app.column_count - 1) {
          app.nextPage();
        } else if (index < app.page_start_index) {
          app.prevPage();
        }
        app.updateDetail();
      }
    },
    setDetailId: function(id) {
      app.current_id = id;
      app.updateDetail();
    },
    nextPage: function(event) {
      if (app.page_index < app.page_count - 1) {
        app.page_index += 1;
        app.page_start_index += app.row_count * app.column_count;
        app.updatePage();
      }
    },
    prevPage: function(event) {
      if (app.page_index > 0) {
        app.page_index -= 1;
        app.page_start_index -= app.row_count * app.column_count;
        app.updatePage();
      }
    },
    setPage: function(index) {
      app.page_index = index;
      app.page_start_index = app.page_index * app.row_count * app.column_count;
      app.updatePage();
    },
    deleteAction: function() {
      $.post("/api/delete/" + app.current_id, { database: app.database }, (data) => {
        let index = app.overview.findIndex((e) => { return e.id == app.current_id; });
        app.overview.splice(index, 1);
        app.current_id = app.overview[Math.min(index, app.overview.length - 1)].id;
        app.page_start_index = app.page_index * app.row_count * app.column_count;
        app.updatePage();
        app.updateDetail();
      });
    },
    objectifyGraspResult: function(data) {
      data.action = {
        pose: {
          x: data.action_pose_x,
          y: data.action_pose_y,
          z: data.action_pose_z,
          a: data.action_pose_a,
          b: data.action_pose_b,
          c: data.action_pose_c,
          d: data.action_pose_d,
        },
        found: data.action_found,
        prob: data.action_prob,
        prob_std: data.action_probstd,
        method: data.action_method,
      };
      data.final = {
        x: data.final_x,
        y: data.final_y,
        z: data.final_z,
        a: data.final_a,
        b: data.final_b,
        c: data.final_c,
        d: data.final_d,
      };
      return data;
    },
    restoreModel: function() {
      $.post("http://127.0.0.1:3000/api/restore-model", { }, (data) => { });
    }
  }
})

var socket = io();

app.row_count = localStorage.getItem('row_count') || app.row_count;
app.column_count = localStorage.getItem('column_count') || app.column_count;
app.draw_bin = localStorage.getItem('draw_bin') || app.draw_bin;

$.get("/api/database-list",(data) => {
  app.database_list = data.map(x => { return {value: x, name: x.split("data/")[1]}; });
  app.database = localStorage.getItem('database') || data[0];
  app.loadDatabase();
});

socket.on('new-result', function (data) {
  app.last_action = app.objectifyGraspResult(data);

  if (data.database == app.database) {
    app.overview.unshift(data);
    app.page_count = Math.ceil(app.overview.length / (app.row_count * app.column_count));
    app.updatePage();
    app.updateDetail();

    $.get("/api/stats", {database: app.database}, (data) => {
      app.stats = data;
    });
  }
});

socket.on('new-attempt', function (data) {
  app.last_action = data;
});

$("body").keydown(function(e) {
  if(e.keyCode == 37) { // left
    app.diffIndex(-1);
  } else if(e.keyCode == 38) { // up
    app.diffIndex(-app.column_count);
  } else if(e.keyCode == 39) { // right
    app.diffIndex(1);
  } else if(e.keyCode == 40) { // down
    app.diffIndex(app.column_count);
  }
});

$('body').bind('wheel', function (event) {
  if (event.originalEvent.deltaY < 0) {
    app.diffIndex(-1);
  } else {
    app.diffIndex(1);
  }
});
