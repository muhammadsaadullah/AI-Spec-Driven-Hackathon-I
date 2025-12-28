const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Bridging the gap between digital AI and physical robots',
  url: 'https://muhammadsaadullah.github.io',
  baseUrl: '/AI-Spec-Driven-Hackathon-I/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'muhammadsaadullah', // Your GitHub username
  projectName: 'AI-Spec-Driven-Hackathon-I', // Your repository name

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
      metadata: [
        {"name": "algolia-site-verification", "content": "D383E01B8119E788"},
      ],
      scripts: [
        {
          src: '/search-injector.js',
          async: true,
        },
      ],
      docs: {
        // Options for local search
        sidebar: {
          hideable: true,
        },
      },
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Home',
          },
          {
            type: 'dropdown',
            label: 'Sections',
            position: 'left',
            items: [
              {
                type: 'doc',
                docId: 'modules/index',
                label: 'Modules',
              },
              {
                type: 'doc',
                docId: 'weekly-breakdown/weeks-1-2-intro-physical-ai',
                label: 'Weekly Breakdown',
              },
              {
                type: 'doc',
                docId: 'assessments/index',
                label: 'Assessments',
              },
              {
                type: 'doc',
                docId: 'reference/glossary',
                label: 'Reference',
              },
            ],
          },
          {
            type: 'search',
            position: 'right',
          },
          {
            href: 'https://github.com/muhammadsaadullah/AI-Spec-Driven-Hackathon-I',
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
                to: '/docs/modules/ros2/',
              },
              {
                label: 'Gazebo Simulation',
                to: '/docs/modules/gazebo-unity/',
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
                href: 'https://github.com/muhammadsaadullah/AI-Spec-Driven-Hackathon-I',
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
