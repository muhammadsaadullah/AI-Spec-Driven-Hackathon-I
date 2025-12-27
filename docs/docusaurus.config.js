const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Bridging the gap between digital AI and physical robots',
  url: 'https://physical-ai-course.github.io',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'physical-ai-course', // Usually your GitHub org/user name.
  projectName: 'book', // Usually your repo name.

  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/facebook/docusaurus/edit/main/website/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/main/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Home',
          },
          {
            type: 'doc',
            docId: 'modules/index',
            position: 'left',
            label: 'Modules',
          },
          {
            type: 'doc',
            docId: 'weekly-breakdown/weeks-1-2-intro-physical-ai',
            position: 'left',
            label: 'Weekly',
          },
          {
            type: 'doc',
            docId: 'assessments/index',
            position: 'left',
            label: 'Assessments',
          },
          {
            type: 'doc',
            docId: 'reference/glossary',
            position: 'left',
            label: 'Reference',
          },
          {
            href: 'https://github.com/physical-ai-course/book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Courses',
            items: [
              {
                label: 'Introduction to Physical AI',
                to: '/docs/intro',
              },
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/modules/ros2',
              },
              {
                label: 'Gazebo Simulation',
                to: '/docs/modules/gazebo-unity',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'Hardware Requirements',
                to: '/docs/hardware-requirements',
              },
              {
                label: 'Glossary',
                to: '/docs/reference/glossary',
              },
              {
                label: 'Notation Guide',
                to: '/docs/reference/notation',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/physical-ai-course/book',
              },
              {
                label: 'NVIDIA Isaac',
                href: 'https://developer.nvidia.com/isaac',
              },
              {
                label: 'ROS 2 Documentation',
                href: 'https://docs.ros.org/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Course. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
});
