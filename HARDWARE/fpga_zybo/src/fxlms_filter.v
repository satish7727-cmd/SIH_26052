module fxlms_filter #(
    parameter TAPS = 32
)(
    input  wire clk,
    input  wire reset,
    input  wire sample_valid,
    input  wire signed [23:0] x_in,
    input  wire signed [23:0] e_in,
    input  wire [31:0] step_mu,
    output reg  signed [23:0] y_out
);
    reg signed [23:0] w [0:TAPS-1];
    reg signed [23:0] x_pipeline [0:TAPS-1];
    integer i;

    always @(posedge clk) begin
        if (reset) begin
            y_out <= 0;
            for (i=0; i<TAPS; i=i+1) begin
                w[i] <= 0;
                x_pipeline[i] <= 0;
            end
        end else if (sample_valid) begin
            // Shift Delay Line
            for (i=TAPS-1; i>0; i=i-1) x_pipeline[i] <= x_pipeline[i-1];
            x_pipeline[0] <= x_in;

            // Compute Filter Output y(n)
            y_out <= 0;
            for (i=0; i<TAPS; i=i+1) begin
                y_out <= y_out + ((w[i] * x_pipeline[i]) >>> 23);
            end

            // Weight Update: w(n+1) = w(n) + mu * e(n) * x(n)
            for (i=0; i<TAPS; i=i+1) begin
                w[i] <= w[i] + (((step_mu * e_in * x_pipeline[i]) >>> 31));
            end
        end
    end
endmodule
