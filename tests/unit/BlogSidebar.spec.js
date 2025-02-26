import { shallowMount } from '@vue/test-utils';
import BlogSidebar from '@/components/BlogSidebar.vue';

describe('BlogSidebar.vue', () => {
  let wrapper;
  beforeEach(() => {
    wrapper = shallowMount(BlogSidebar, {
      methods: {
        // Stub getBlogSections to return minimal test data
        getBlogSections() {
          return [
            { name: 'section1', expanded: false, blogs: [{ title: 'blog1', content: 'content' }] }
          ];
        }
      }
    });
  });

  it('initially is not collapsed', () => {
    expect(wrapper.vm.isMenuCollapsed).toBe(false);
    expect(wrapper.find('.blog-list').isVisible()).toBe(true);
  });

  it('toggles menu collapse on hamburger click', async () => {
    const hamburger = wrapper.find('.hamburger');
    await hamburger.trigger('click');
    expect(wrapper.vm.isMenuCollapsed).toBe(true);
    expect(wrapper.find('.blog-list').isVisible()).toBe(false);

    await hamburger.trigger('click');
    expect(wrapper.vm.isMenuCollapsed).toBe(false);
    expect(wrapper.find('.blog-list').isVisible()).toBe(true);
  });

  it('emits page-selected when clicking About Me', async () => {
    const aboutItem = wrapper.find('li');
    await aboutItem.trigger('click');
    expect(wrapper.emitted('page-selected')).toBeTruthy();
  });
});
