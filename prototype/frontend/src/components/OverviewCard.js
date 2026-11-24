import React from 'react';

const OverviewCard = ({ config, env }) => {
  return (
    <div className="card">
      <h3>Experiment Configuration</h3>
      <pre>{JSON.stringify(config, null, 2)}</pre>
    </div>
  );
};

export default OverviewCard;