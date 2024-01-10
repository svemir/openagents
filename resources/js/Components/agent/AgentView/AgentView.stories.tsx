import type { Meta, StoryObj } from '@storybook/react';
import { AgentView } from '.';
import { Navbar } from '@/Components/nav/Navbar';
import { demoAgent } from '../../../../../agentgraph/components/Node/Node.demodata';
import { demoUser } from '@/lib/dummyData';

const meta = {
  title: 'OpenAgents/AgentView',
  component: AgentView,
  parameters: { layout: 'fullscreen' },
  argTypes: {},
  decorators: [
    (Story) => (
      <>
        <Navbar user={demoUser} />
        <div className="h-screen pt-16">
          <Story />
        </div>
      </>
    ),
  ],
} satisfies Meta<typeof AgentView>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    agent: demoAgent,
    conversation: {
      id: 1
    },
    files: [],
    owner: 'DemoMan'
  }
}
