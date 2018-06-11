export default {
  name: 'PageHeader',
  data: function () {
    return {
      showNav: false
    };
  },
  props: {
    title: String,
    onMenuClick: Function
  }
};
