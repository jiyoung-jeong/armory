// Auto-generated from _summary_new_5min_b3/results_max_batch_size=3.csv
// Per-tier throughput (successes/min) and starvation (percent) vs number of robots.

export const ROBOTS = [2, 4, 6, 8, 10];
export interface Scenario {
  key: string;
  title: string;
  subtitle: string;
  thrMin: number;
  thrMax: number;
  thrTicks: number[];
  starvMax: number;
  starvTicks: number[];
  series: {
    fastThr: Record<string, number[]>;
    slowThr: Record<string, number[]>;
    fastStarv: Record<string, number[]>;
    slowStarv: Record<string, number[]>;
  };
}

export const ONE_FAST: Scenario = {
  "key": "1f9s",
  "title": "One Fast (1 fast, 9 slow)",
  "subtitle": "LIBERO-10 suite, max batch size 3",
  "thrMin": 2.5,
  "thrMax": 5.2,
  "thrTicks": [
    3,
    4,
    5
  ],
  "starvMax": 40,
  "starvTicks": [
    0,
    10,
    20,
    30,
    40
  ],
  "series": {
    "fastThr": {
      "EDF": [
        3.85,
        3.89,
        3.34,
        3.42,
        3.12
      ],
      "RR": [
        4.15,
        3.73,
        3.25,
        3.18,
        2.66
      ],
      "LA": [
        4.06,
        3.69,
        3.2,
        3.13,
        3.27
      ],
      "LA@3": [
        4.1,
        3.93,
        3.53,
        3.37,
        3.2
      ],
      "LA@5": [
        3.59,
        3.99,
        3.78,
        3.72,
        3.71
      ]
    },
    "slowThr": {
      "EDF": [
        4.98,
        4.77,
        4.71,
        4.29,
        4.11
      ],
      "RR": [
        4.97,
        4.83,
        4.76,
        4.41,
        4.2
      ],
      "LA": [
        4.5,
        4.77,
        4.68,
        4.36,
        4.09
      ],
      "LA@3": [
        5.01,
        4.83,
        4.46,
        4.13,
        4.04
      ],
      "LA@5": [
        4.74,
        4.73,
        4.35,
        4.1,
        3.81
      ]
    },
    "fastStarv": {
      "EDF": [
        0.94,
        10.4,
        22.75,
        25.27,
        27.99
      ],
      "RR": [
        0.99,
        10.79,
        22.84,
        34.63,
        38.27
      ],
      "LA": [
        1.41,
        12.95,
        23.32,
        27.22,
        31.61
      ],
      "LA@3": [
        0.94,
        6.18,
        19.68,
        23.82,
        25.1
      ],
      "LA@5": [
        0.86,
        2.38,
        5.88,
        11.76,
        15.31
      ]
    },
    "slowStarv": {
      "EDF": [
        1.16,
        1.62,
        2.09,
        8.55,
        15.57
      ],
      "RR": [
        1.08,
        1.67,
        2.14,
        10.51,
        16.01
      ],
      "LA": [
        1.3,
        1.5,
        2.5,
        8.65,
        15.13
      ],
      "LA@3": [
        1.1,
        1.89,
        3.41,
        10.35,
        17.23
      ],
      "LA@5": [
        1.14,
        2.5,
        8.75,
        14.91,
        22.02
      ]
    }
  }
};

export const HALF_FAST: Scenario = {
  "key": "5f5s",
  "title": "Half Fast (5 fast, 5 slow)",
  "subtitle": "LIBERO-10 suite, max batch size 3",
  "thrMin": 0,
  "thrMax": 5.2,
  "thrTicks": [
    0,
    1,
    2,
    3,
    4,
    5
  ],
  "starvMax": 80,
  "starvTicks": [
    0,
    20,
    40,
    60,
    80
  ],
  "series": {
    "fastThr": {
      "EDF": [
        3.78,
        4.26,
        3.72,
        3.47,
        3.2
      ],
      "RR": [
        3.89,
        4.31,
        3.28,
        3.18,
        2.97
      ],
      "LA": [
        4.03,
        4.22,
        3.63,
        3.51,
        2.92
      ],
      "LA@3": [
        4.11,
        4.61,
        3.95,
        3.84,
        3.43
      ],
      "LA@5": [
        3.82,
        4.43,
        4.07,
        3.97,
        3.73
      ]
    },
    "slowThr": {
      "EDF": [
        4.96,
        4.41,
        4.49,
        4.16,
        3.76
      ],
      "RR": [
        4.85,
        4.5,
        4.59,
        4.34,
        4.22
      ],
      "LA": [
        4.71,
        4.43,
        4.72,
        4.17,
        3.98
      ],
      "LA@3": [
        4.93,
        4.65,
        4.53,
        3.88,
        3.59
      ],
      "LA@5": [
        4.78,
        2.78,
        3.48,
        2.68,
        0.49
      ]
    },
    "fastStarv": {
      "EDF": [
        0.88,
        10.63,
        22.97,
        26.86,
        32.95
      ],
      "RR": [
        0.93,
        10.3,
        23.11,
        34.91,
        38.38
      ],
      "LA": [
        0.97,
        13.29,
        23.09,
        28.32,
        34.08
      ],
      "LA@3": [
        0.98,
        7.44,
        18.55,
        24.65,
        30.27
      ],
      "LA@5": [
        0.94,
        2.5,
        13.12,
        18.57,
        21.88
      ]
    },
    "slowStarv": {
      "EDF": [
        1.07,
        1.53,
        2.06,
        13.48,
        20.47
      ],
      "RR": [
        1.08,
        1.53,
        2.09,
        10.49,
        16.03
      ],
      "LA": [
        1.05,
        1.26,
        2.18,
        11.52,
        19.74
      ],
      "LA@3": [
        1.08,
        6.39,
        11.29,
        20.86,
        29.82
      ],
      "LA@5": [
        1.09,
        40.38,
        34.44,
        49.25,
        74.71
      ]
    }
  }
};
