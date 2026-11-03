export interface PredictionItem {
  label: string;
  icon: string;
  confidence: number;
}

export interface ModelInfo {
  name: string;
  trained: boolean;
  classes: number;
}

export interface PredictionResponse {
  success: boolean;
  is_mock?: boolean;
  latency_ms?: number;
  message?: string;
  top_prediction?: PredictionItem;
  predictions?: PredictionItem[];
  model_info?: ModelInfo;
  error?: string;
}

export interface SampleImage {
  id: string;
  title: string;
  icon: string;
  category: string;
  description: string;
  dataUrl: string;
}

