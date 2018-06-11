import PageHeader from '@/components/PageHeader';

export default {
  name: 'PageContent',
  components: {
    PageHeader,
  },
  props: {
    pageTitle: String
  },
  mounted: function () {
    window.onscroll = function () {
      function myFunction () {
        const header = this.$refs.header;
        const sticky = header.offsetTop;
        if (window.pageYOffset >= sticky) {
          header.classList.add('sticky');
        } else {
          header.classList.remove('sticky');
        }
      }
    };
  },
};
