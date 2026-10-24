module fast_feature_extractor (
    input  wire clk,
    input  wire reset,
    input  wire signed [23:0] audio_in,
    input  wire sample_valid,
    output reg  [31:0] rms_energy,
    output reg  [23:0] peak_val,
    output reg  fast_impulse_detect
);
    reg signed [47:0] energy_acc;
    reg [15:0] sample_cnt;

    always @(posedge clk) begin
        if (reset) begin
            energy_acc <= 0;
            sample_cnt <= 0;
            peak_val <= 0;
            fast_impulse_detect <= 0;
        end else if (sample_valid) begin
            energy_acc <= energy_acc + (audio_in * audio_in);
            if ($abs(audio_in) > peak_val) peak_val <= $abs(audio_in);
            
            sample_cnt <= sample_cnt + 1;
            if (sample_cnt == 512) begin
                rms_energy <= energy_acc[47:16];
                // Ultra-Fast Impulse Threshold Trigger
                if (peak_val > 24'sd4194304) fast_impulse_detect <= 1;
                else fast_impulse_detect <= 0;
                
                energy_acc <= 0;
                sample_cnt <= 0;
                peak_val <= 0;
            end
        end
    end
endmodule
