import * as React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import MuiInput from '@mui/material/Input';
import { setGlobalState, useGlobalState } from '../states';


const Input = styled(MuiInput)`
  width: 42px;
`;

export default function SliderBar() {
 
  const handleSliderChange = (event, newValue) => {
    setGlobalState("value",newValue)
  };

  const handleInputChange = (event) => {
    setGlobalState(event.target.value === '' ? '' : Number(event.target.value));
  };

  const handleBlur = () => {
    if (value < 0) {
      setGlobalState("value",0);
    } else if (value > 10) {
      setGlobalState("value",10);
    }
  };
  const [value]=useGlobalState("value")
  return (
    <Box sx={{ width: 250 }}>
      <Typography id="input-slider" gutterBottom>
      </Typography>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs>
          <Slider
            value={typeof value === 'number' ? value : 0}
            onChange={handleSliderChange}
            aria-labelledby="input-slider"
            min={1}
            max={10}
          />
        </Grid>
        <Grid item>
          <Input
            value={value}
            size="small"
            onChange={handleInputChange}
            onBlur={handleBlur}
            inputProps={{
              step: 1,
              min: 1,
              max: 10,
              type: 'number',
              'aria-labelledby': 'input-slider',
            }}
          />
        </Grid>
      </Grid>
    </Box>
  );
}
